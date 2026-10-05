/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::repr::{Guard, Representation, Set};
use diskann::graph::AdjacencyList;
use diskann_quantization::{
    algorithms::transforms::{TargetDim, TransformKind},
    spherical::{PreScale, SphericalQuantizer, SupportedMetric},
};
use diskann_utils::views::Matrix;
use rand::{SeedableRng, rngs::StdRng};

fn representation(
    bits: usize,
    metric: SupportedMetric,
    rerank: Rerank,
    capacity: usize,
) -> Spherical {
    let data = Matrix::from_fn(16, 5, |rc| ((rc.row * 7 + rc.col * 3) % 19) as f32 - 9.0);
    let quantizer = SphericalQuantizer::train(
        data.as_view(),
        TransformKind::PaddingHadamard {
            target_dim: TargetDim::Natural,
        },
        metric,
        PreScale::None,
        &mut StdRng::seed_from_u64(42),
        GlobalAllocator,
    )
    .unwrap();
    let quantizer = match bits {
        1 => quantizer.as_quantizer::<1>().unwrap(),
        2 => quantizer.as_quantizer::<2>().unwrap(),
        4 => quantizer.as_quantizer::<4>().unwrap(),
        8 => quantizer.as_quantizer::<8>().unwrap(),
        _ => unreachable!(),
    };
    let mut store_config = store::Config::new();
    store_config
        .epoch_guard_slots(NonZeroUsize::new(3).unwrap())
        .freelist_recycle_capacity(NonZeroU32::new(2).unwrap());
    Spherical::config(
        quantizer,
        Capacity::new(capacity),
        MaxDegree::new(3),
        Matrix::from_fn(2, 5, |rc| data.row(rc.row)[rc.col]),
        rerank,
    )
    .unwrap()
    .store(store_config)
    .prefetch(NonZeroUsize::new(7))
    .build()
    .unwrap()
}

fn save(repr: &mut Spherical) -> Vec<u8> {
    let mut bytes = Vec::new();
    repr.write_snapshot(&mut bytes).unwrap();
    bytes
}

fn vectors(repr: &Spherical, id: u32) -> (Vec<u8>, Option<Vec<u8>>) {
    repr.store
        .guard(|slots, guard| {
            let compressed = slots
                .first()
                .reader(guard.share())
                .read(id as usize)
                .unwrap()
                .to_vec();
            let rerank = slots
                .second()
                .slots()
                .map(|s| s.reader(guard).read(id as usize).unwrap().to_vec());
            (compressed, rerank)
        })
        .unwrap()
}

#[test]
fn roundtrip_all_bits_metrics_and_reranking() {
    for bits in [1, 2, 4, 8] {
        for metric in [
            SupportedMetric::SquaredL2,
            SupportedMetric::InnerProduct,
            SupportedMetric::Cosine,
        ] {
            for rerank in [Rerank::None, Rerank::F16, Rerank::F32] {
                let mut original = representation(bits, metric, rerank, 4);
                let first = Set::set(&original, &[1.0, -0.0, 3.25, -2.0, 0.5][..]).unwrap();
                let id = first.id();
                first.publish();
                let deleted = Set::set(&original, &[2.0; 5][..]).unwrap();
                let deleted_id = deleted.id();
                deleted.publish();
                original.retire(deleted_id).unwrap();
                original
                    .store
                    .neighbors()
                    .set(id, &[5, deleted_id, 4])
                    .unwrap();
                original.store.neighbors().set(4, &[id]).unwrap();
                original.store.neighbors().set(deleted_id, &[5]).unwrap();
                let bytes = save(&mut original);
                let mut loaded = Spherical::read_snapshot(&mut bytes.as_slice()).unwrap();
                assert_eq!(loaded.quantizer.metric(), metric);
                assert_eq!(loaded.quantizer.nbits(), bits);
                assert_eq!(loaded.quantizer.dim(), 8);
                assert_eq!(loaded.dim(), 5);
                assert_eq!(loaded.lookahead, original.lookahead);
                assert_eq!(loaded.store.config(), original.store.config());
                assert_eq!(loaded.is_readable(deleted_id), Some(false));
                for i in [id, 4, 5] {
                    assert_eq!(vectors(&original, i), vectors(&loaded, i));
                }
                let mut neighbors = AdjacencyList::new();
                loaded.store.neighbors().get(id, &mut neighbors).unwrap();
                assert_eq!(&*neighbors, &[5, deleted_id, 4]);
                assert_eq!(save(&mut loaded), bytes);
            }
        }
    }
}

#[test]
fn empty_sparse_and_full_occupancy() {
    for capacity in [0, 4] {
        for count in 0..=capacity {
            let mut original = representation(4, SupportedMetric::SquaredL2, Rerank::F16, capacity);
            for _ in 0..count {
                Set::set(&original, &[1.0; 5][..]).unwrap().publish();
            }
            let bytes = save(&mut original);
            let mut loaded = Spherical::read_snapshot(&mut bytes.as_slice()).unwrap();
            assert_eq!(save(&mut loaded), bytes);
            for _ in count..capacity {
                Set::set(&loaded, &[2.0; 5][..]).unwrap().publish();
            }
            assert!(Set::set(&loaded, &[2.0; 5][..]).is_err());
        }
    }
}

#[test]
fn growth_relocates_frozen_vectors_and_edges() {
    for rerank in [Rerank::None, Rerank::F16, Rerank::F32] {
        let mut original = representation(2, SupportedMetric::InnerProduct, rerank, 4);
        Set::set(&original, &[1.0; 5][..]).unwrap().publish();
        let deleted = Set::set(&original, &[2.0; 5][..]).unwrap();
        let deleted_id = deleted.id();
        deleted.publish();
        original.retire(deleted_id).unwrap();
        original
            .store
            .neighbors()
            .set(0, &[5, deleted_id, 4])
            .unwrap();
        original.store.neighbors().set(4, &[0, 5]).unwrap();
        original.store.neighbors().set(deleted_id, &[5]).unwrap();
        let bytes = save(&mut original);
        let loaded = Spherical::read_snapshot_with_options(
            &mut bytes.as_slice(),
            LoadOptions {
                capacity: Some(8),
                epoch_guard_slots: NonZeroUsize::new(5),
                prefetch: Some(None),
            },
        )
        .unwrap();
        assert_eq!(loaded.capacity(), Capacity::new(8));
        assert_eq!(loaded.lookahead, None);
        assert_eq!(loaded.store.config().snapshot_parameters().0.get(), 5);
        assert_eq!(loaded.store.config().snapshot_parameters().1.get(), 2);
        for (before, after) in [(0, 0), (4, 8), (5, 9)] {
            assert_eq!(vectors(&original, before), vectors(&loaded, after));
        }
        let mut neighbors = AdjacencyList::new();
        for (id, expected) in [
            (0, vec![9, deleted_id, 8]),
            (8, vec![0, 9]),
            (deleted_id, vec![9]),
        ] {
            loaded.store.neighbors().get(id, &mut neighbors).unwrap();
            assert_eq!(&*neighbors, expected);
        }
        for _ in 1..8 {
            Set::set(&loaded, &[3.0; 5][..]).unwrap().publish();
        }
        assert!(Set::set(&loaded, &[3.0; 5][..]).is_err());
        for prefetch in [None, Some(NonZeroUsize::new(9))] {
            let loaded = Spherical::read_snapshot_with_options(
                &mut bytes.as_slice(),
                LoadOptions {
                    capacity: Some(4),
                    prefetch,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(loaded.lookahead, prefetch.unwrap_or(original.lookahead));
        }
    }
}

fn header(bytes: &[u8]) -> (Header, usize) {
    let mut cursor = std::io::Cursor::new(bytes);
    let header = persistence::read(&mut cursor).unwrap();
    (header, cursor.position() as usize)
}

fn with_header(bytes: &[u8], change: impl FnOnce(&mut Header)) -> Vec<u8> {
    let (mut header, end) = header(bytes);
    change(&mut header);
    let mut result = Vec::new();
    persistence::write(&mut result, &header).unwrap();
    result.extend_from_slice(&bytes[end..]);
    result
}

#[test]
fn reject_invalid_headers_quantizers_and_growth() {
    let bytes = save(&mut representation(
        4,
        SupportedMetric::SquaredL2,
        Rerank::F16,
        4,
    ));
    for change in [
        |h: &mut Header| h.representation = 1,
        |h: &mut Header| h.layout = 2,
        |h: &mut Header| h.rerank = 3,
        |h: &mut Header| h.nbits = 3,
        |h: &mut Header| h.frozen = 0,
        |h: &mut Header| h.capacity = u32::MAX,
        |h: &mut Header| h.full_dim = u64::MAX,
        |h: &mut Header| h.dim = u64::MAX,
        |h: &mut Header| h.bytes += 1,
        |h: &mut Header| h.full_dim += 1,
        |h: &mut Header| {
            h.nbits = 2;
            h.bytes = 8;
        },
        |h: &mut Header| h.epoch_guard_slots = 0,
        |h: &mut Header| h.epoch_guard_slots = u64::MAX,
        |h: &mut Header| h.freelist_recycle_capacity = 0,
        |h: &mut Header| h.quantizer_len = 0,
        |h: &mut Header| h.quantizer_len = 7,
        |h: &mut Header| h.quantizer_len = u64::MAX,
    ] {
        assert!(Spherical::read_snapshot(&mut with_header(&bytes, change).as_slice()).is_err());
    }
    for capacity in [3, u32::MAX] {
        assert!(
            Spherical::read_snapshot_with_options(
                &mut bytes.as_slice(),
                LoadOptions {
                    capacity: Some(capacity),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    let (_, start) = header(&bytes);
    for offset in [0, 4] {
        let mut corrupt = bytes.clone();
        corrupt[start + offset..start + offset + 4].fill(255);
        assert!(Spherical::read_snapshot(&mut corrupt.as_slice()).is_err());
    }
    // Loading either representation as the other must fail.
    assert!(crate::repr::Full::<f32>::read_snapshot(&mut bytes.as_slice()).is_err());
    let mut full = crate::repr::Full::config(
        Capacity::new(4),
        MaxDegree::new(3),
        diskann_vector::distance::Metric::L2,
        Matrix::row_vector(vec![1.0f32; 5].into()),
    )
    .unwrap()
    .build()
    .unwrap();
    let mut full_bytes = Vec::new();
    full.write_snapshot(&mut full_bytes).unwrap();
    assert!(Spherical::read_snapshot(&mut full_bytes.as_slice()).is_err());
}

#[test]
fn reject_truncation_occupancy_and_graph_corruption() {
    for rerank in [Rerank::None, Rerank::F16, Rerank::F32] {
        let mut original = representation(4, SupportedMetric::SquaredL2, rerank, 1);
        original.store.neighbors().set(0, &[2]).unwrap();
        let bytes = save(&mut original);
        for end in 0..bytes.len() {
            assert!(
                Spherical::read_snapshot(&mut &bytes[..end]).is_err(),
                "prefix {end}"
            );
        }
        let (h, start) = header(&bytes);
        let slots = start + h.quantizer_len as usize;
        let mut corrupt = bytes.clone();
        corrupt[slots] = 2;
        assert!(Spherical::read_snapshot(&mut corrupt.as_slice()).is_err());
        corrupt = bytes.clone();
        corrupt[slots + 1] = 0; // Missing first frozen point.
        assert!(Spherical::read_snapshot(&mut corrupt.as_slice()).is_err());
        let rerank_bytes = match rerank {
            Rerank::None => 0,
            Rerank::F16 => 10,
            Rerank::F32 => 20,
        };
        let graph = slots + 1 + 2 * (1 + h.bytes as usize + rerank_bytes);
        for (offset, value) in [(0, 4u32), (4, 3u32)] {
            corrupt = bytes.clone();
            corrupt[graph + offset..graph + offset + 4].copy_from_slice(&value.to_le_bytes());
            // The invalid neighbor would fit in the enlarged store, but not the saved store.
            assert!(
                Spherical::read_snapshot_with_options(
                    &mut corrupt.as_slice(),
                    LoadOptions {
                        capacity: Some(8),
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
    }
}

#[test]
fn metadata_and_reranking_words_are_little_endian() {
    let words = [0x8000u16, 0x3c00, 0x1234];
    let native: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let mut encoded = Vec::new();
    write_words::<2, _>(&mut encoded, &native).unwrap();
    assert_eq!(encoded, [0, 128, 0, 60, 52, 18]);
    native_words::<2>(&mut encoded);
    assert_eq!(encoded, native);

    let words = [0x80000000u32, 0x3f800001, 0x12345678];
    let native: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let mut encoded = Vec::new();
    write_words::<4, _>(&mut encoded, &native).unwrap();
    assert_eq!(encoded, [0, 0, 0, 128, 1, 0, 128, 63, 120, 86, 52, 18]);
    native_words::<4>(&mut encoded);
    assert_eq!(encoded, native);
}

#[test]
fn f32_precision_and_reranking_survive_restore() {
    use crate::{counters::Counters, repr::Search};
    use diskann::neighbor::Neighbor;
    use diskann_vector::distance::DistanceProvider;

    let query = [1.0002, -0.0, 3.1252, -2.0003, 0.5001];
    let data = [
        [1.0001, -0.0, 3.1253, -2.0001, 0.5002],
        [1.0003, 0.0, 3.1251, -2.0004, 0.5003],
    ];
    for metric in [
        SupportedMetric::SquaredL2,
        SupportedMetric::InnerProduct,
        SupportedMetric::Cosine,
    ] {
        let mut original = representation(4, metric, Rerank::F32, 3);
        let mut ids = Vec::new();
        for vector in &data {
            let slot = Set::set(&original, vector.as_slice()).unwrap();
            ids.push(slot.id());
            slot.publish();
        }
        let deleted = Set::set(&original, &[2.0; 5][..]).unwrap();
        let deleted_id = deleted.id();
        deleted.publish();
        original.retire(deleted_id).unwrap();

        let bytes = save(&mut original);
        let (h, start) = header(&bytes);
        assert_eq!(h.rerank, 2);
        let first_payload = start + h.quantizer_len as usize + 1 + h.bytes as usize;
        let expected_bytes: Vec<u8> = data[0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&bytes[first_payload..first_payload + 20], expected_bytes);
        let loaded = Spherical::read_snapshot(&mut bytes.as_slice()).unwrap();

        let distance = <f32 as DistanceProvider<f32>>::distance_comparer(
            super::super::convert_metric(metric),
            Some(query.len()),
        );
        let rounded: Vec<f32> = data[0]
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        assert_ne!(data[0].as_slice(), rounded);
        if metric == SupportedMetric::SquaredL2 {
            assert_ne!(
                distance.call(&query, &data[0]),
                distance.call(&query, &rounded)
            );
        }
        let mut expected: Vec<_> = ids
            .iter()
            .zip(&data)
            .map(|(&id, v)| Neighbor::new(id, distance.call(&query, v)))
            .collect();
        expected.sort_unstable_by(diskann::neighbor::ord::fast_distance);

        for repr in [&original, &loaded] {
            for (&id, vector) in ids.iter().zip(&data) {
                assert_eq!(
                    vectors(repr, id).1.unwrap(),
                    bytemuck::cast_slice::<f32, u8>(vector)
                );
            }
            let counters = Counters::new();
            let mut accessor = repr.search_accessor(&query, &(), counters.local()).unwrap();
            let mut candidates: Vec<_> = ids
                .iter()
                .chain([&deleted_id])
                .map(|&id| Neighbor::new(id, -100.0))
                .collect();
            accessor
                .get_post_process()
                .unwrap()
                .post_process(&mut candidates)
                .unwrap();
            assert_eq!(
                candidates.iter().map(|n| n.as_tuple()).collect::<Vec<_>>(),
                expected.iter().map(|n| n.as_tuple()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn reject_f32_reranking_allocation_overflow() {
    let bytes = save(&mut representation(
        4,
        SupportedMetric::SquaredL2,
        Rerank::F32,
        4,
    ));
    // The dimension fits in usize, but the f32 payload size does not.
    let corrupt = with_header(&bytes, |h| h.full_dim = (usize::MAX / 4 + 1) as u64);
    let err = Spherical::read_snapshot(&mut corrupt.as_slice()).unwrap_err();
    assert!(
        err.to_string()
            .contains("snapshot vector or graph allocation overflows")
    );
}
