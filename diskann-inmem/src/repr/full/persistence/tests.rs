/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Cursor;

use diskann::graph::AdjacencyList;
use diskann_utils::views::Matrix;

use super::*;
use crate::repr::{Guard, Representation, Set};

fn roundtrip<T: FullPrecision>(values: &[T])
where
    Full<T>: Snapshot,
{
    for metric in [
        Metric::L2,
        Metric::InnerProduct,
        Metric::Cosine,
        Metric::CosineNormalized,
    ] {
        for lookahead in [None, NonZeroUsize::new(7)] {
            let mut config = store::Config::new();
            config
                .epoch_guard_slots(NonZeroUsize::new(3).unwrap())
                .freelist_recycle_capacity(NonZeroU32::new(2).unwrap());
            let mut original = Full::config(
                Capacity::new(4),
                MaxDegree::new(3),
                metric,
                Matrix::from_fn(2, values.len(), |rc| values[rc.col]),
            )
            .unwrap()
            .store(config.clone())
            .prefetch(lookahead)
            .build()
            .unwrap();
            let first = original.set(values).unwrap();
            let first_id = first.id();
            first.publish();
            let second = original.set(values).unwrap();
            let deleted_id = second.id();
            second.publish();
            original.retire(deleted_id).unwrap();
            // Keep an edge to a retired slot and non-sorted neighbor order.
            original
                .store
                .neighbors()
                .set(first_id, &[5, deleted_id, 4])
                .unwrap();
            original.store.neighbors().set(4, &[first_id]).unwrap();
            original.store.neighbors().set(deleted_id, &[5]).unwrap();
            let mut bytes = Vec::new();
            original.write_snapshot(&mut bytes).unwrap();
            let mut loaded = Full::<T>::read_snapshot(&mut bytes.as_slice()).unwrap();
            assert_eq!(loaded.metric(), metric);
            assert_eq!(loaded.dim(), values.len());
            assert_eq!(loaded.capacity(), original.capacity());
            assert_eq!(loaded.id_limit(), original.id_limit());
            assert_eq!(loaded.max_degree(), original.max_degree());
            assert_eq!(loaded.lookahead, lookahead);
            assert_eq!(loaded.store.config(), &config);
            assert_eq!(loaded.is_readable(deleted_id), Some(false));
            for id in [first_id, 4, 5] {
                assert_eq!(
                    bytemuck::cast_slice::<T, u8>(&loaded.get(id).unwrap()),
                    bytemuck::cast_slice::<T, u8>(values)
                );
            }
            let mut neighbors = AdjacencyList::new();
            loaded
                .store
                .neighbors()
                .get(first_id, &mut neighbors)
                .unwrap();
            assert_eq!(&*neighbors, &[5, deleted_id, 4]);
            let mut again = Vec::new();
            loaded.write_snapshot(&mut again).unwrap();
            assert_eq!(again, bytes);

            let grown = Full::<T>::read_snapshot_with_options(
                &mut bytes.as_slice(),
                LoadOptions {
                    capacity: Some(8),
                    epoch_guard_slots: NonZeroUsize::new(5),
                    prefetch: Some(None),
                },
            )
            .unwrap();
            assert_eq!(grown.capacity(), Capacity::new(8));
            assert_eq!(grown.store.frozen().len(), 2);
            assert_eq!(grown.lookahead, None);
            assert_eq!(grown.store.config().snapshot_parameters().0.get(), 5);
            assert_eq!(grown.is_readable(deleted_id), Some(false));
            for id in [first_id, 8, 9] {
                assert_eq!(
                    bytemuck::cast_slice::<T, u8>(&grown.get(id).unwrap()),
                    bytemuck::cast_slice::<T, u8>(values)
                );
            }
            for (id, expected) in [
                (first_id, vec![9, deleted_id, 8]),
                (8, vec![first_id]),
                (deleted_id, vec![9]),
            ] {
                grown.store.neighbors().get(id, &mut neighbors).unwrap();
                assert_eq!(&*neighbors, expected.as_slice());
            }
            // All deleted and newly allocated writable slots must be reusable.
            for _ in 1..8 {
                let slot = grown.set(values).unwrap();
                assert!(slot.id() < 8);
                slot.publish();
            }
            assert!(grown.set(values).is_err());
        }
    }
}

#[test]
fn all_scalars_metrics_and_tuning() {
    roundtrip(&[f32::from_bits(0x7fc0_1234), -0.0, f32::INFINITY, -2.5]);
    roundtrip(&[
        half::f16::from_bits(0x7e13),
        half::f16::NEG_ZERO,
        half::f16::INFINITY,
    ]);
    roundtrip(&[0u8, 255, 17]);
    roundtrip(&[i8::MIN, i8::MAX, -7]);
}

#[test]
fn zero_capacity_and_start_points() {
    for (capacity, frozen, dim) in [(0, 0, 3), (0, 2, 3), (2, 0, 3)] {
        let mut original = Full::config(
            Capacity::new(capacity),
            MaxDegree::new(0),
            Metric::L2,
            Matrix::from_element(frozen, dim, 0f32),
        )
        .unwrap()
        .build()
        .unwrap();
        let mut bytes = Vec::new();
        original.write_snapshot(&mut bytes).unwrap();
        let loaded = Full::<f32>::read_snapshot(&mut bytes.as_slice()).unwrap();
        assert_eq!(loaded.capacity(), Capacity::new(capacity));
        assert_eq!(loaded.store.frozen().len(), frozen);
        assert_eq!(loaded.dim(), dim);
    }
}

fn sample() -> Vec<u8> {
    let mut full = Full::config(
        Capacity::new(1),
        MaxDegree::new(2),
        Metric::L2,
        Matrix::row_vector(vec![3.5f32].into()),
    )
    .unwrap()
    .build()
    .unwrap();
    full.set(&[1.25][..]).unwrap().publish();
    full.store.neighbors().set(0, &[1]).unwrap();
    let mut bytes = Vec::new();
    full.write_snapshot(&mut bytes).unwrap();
    bytes
}

fn with_header(bytes: &[u8], change: fn(&mut Header)) -> Vec<u8> {
    let mut cursor = Cursor::new(bytes);
    let mut header: Header = persistence::read(&mut cursor).unwrap();
    change(&mut header);
    let mut result = Vec::new();
    persistence::write(&mut result, &header).unwrap();
    result.extend_from_slice(&bytes[cursor.position() as usize..]);
    result
}

#[test]
fn reject_invalid_header_before_allocating() {
    let bytes = sample();
    let corruptions: &[fn(&mut Header)] = &[
        |h| h.representation = 2,
        |h| h.scalar = 0,
        |h| h.metric = -1,
        |h| h.epoch_guard_slots = 0,
        |h| h.epoch_guard_slots = u64::MAX,
        |h| h.freelist_recycle_capacity = 0,
        |h| h.dim = u64::MAX,
        |h| {
            h.capacity = u32::MAX;
            h.frozen = 1;
        },
        |h| {
            h.capacity = u32::MAX - 1;
            h.max_degree = u32::MAX;
        },
    ];
    for change in corruptions {
        let changed = with_header(&bytes, *change);
        assert!(Full::<f32>::read_snapshot(&mut changed.as_slice()).is_err());
    }
    assert!(Full::<i8>::read_snapshot(&mut bytes.as_slice()).is_err());
}

#[test]
fn reject_truncation_occupancy_and_invalid_edges() {
    let bytes = sample();
    for end in 0..bytes.len() {
        assert!(
            Full::<f32>::read_snapshot(&mut &bytes[..end]).is_err(),
            "offset {end}"
        );
    }
    let mut cursor = Cursor::new(&bytes);
    let _: Header = persistence::read(&mut cursor).unwrap();
    let header_end = cursor.position() as usize;
    assert_eq!(header_end, 52);
    let mut changed = bytes.clone();
    changed[header_end] = 2; // Invalid boolean.
    assert!(Full::<f32>::read_snapshot(&mut changed.as_slice()).is_err());
    let mut changed = bytes.clone();
    changed[header_end + 5] = 0; // Missing frozen vector.
    assert!(Full::<f32>::read_snapshot(&mut changed.as_slice()).is_err());
    let graph = header_end + 2 * 5;
    let mut changed = bytes.clone();
    changed[graph..graph + 4].copy_from_slice(&3u32.to_le_bytes());
    assert!(Full::<f32>::read_snapshot(&mut changed.as_slice()).is_err());
    let mut changed = bytes;
    changed[graph + 4..graph + 8].copy_from_slice(&2u32.to_le_bytes());
    assert!(Full::<f32>::read_snapshot(&mut changed.as_slice()).is_err());
}

#[test]
fn vector_payload_is_little_endian() {
    let bytes = sample();
    assert_eq!(&bytes[52..62], &[1, 0, 0, 160, 63, 1, 0, 0, 96, 64]);
}

#[test]
fn reject_invalid_growth_and_saved_edges() {
    let bytes = sample();
    for capacity in [0, u32::MAX] {
        assert!(
            Full::<f32>::read_snapshot_with_options(
                &mut bytes.as_slice(),
                LoadOptions {
                    capacity: Some(capacity),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    let mut changed = bytes;
    // This edge would be valid in the enlarged store, but was invalid when saved.
    changed[66..70].copy_from_slice(&2u32.to_le_bytes());
    assert!(
        Full::<f32>::read_snapshot_with_options(
            &mut changed.as_slice(),
            LoadOptions {
                capacity: Some(8),
                ..Default::default()
            }
        )
        .is_err()
    );
}
