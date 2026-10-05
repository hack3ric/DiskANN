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
