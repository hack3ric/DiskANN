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
            for rerank in [Rerank::None, Rerank::F16] {
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
    for rerank in [Rerank::None, Rerank::F16] {
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
        |h: &mut Header| h.rerank = 2,
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
    for rerank in [Rerank::None, Rerank::F16] {
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
        let graph =
            slots + 1 + 2 * (1 + h.bytes as usize + if rerank == Rerank::F16 { 10 } else { 0 });
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
    write_words(&mut encoded, &native).unwrap();
    assert_eq!(encoded, [0, 128, 0, 60, 52, 18]);
    native_words(&mut encoded);
    assert_eq!(encoded, native);
}
