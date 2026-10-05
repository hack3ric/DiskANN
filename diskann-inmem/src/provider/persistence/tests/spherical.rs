/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::repr::{Spherical, spherical::Rerank};
use diskann_quantization::{
    algorithms::transforms::TransformKind,
    alloc::GlobalAllocator,
    spherical::{PreScale, SphericalQuantizer, SupportedMetric},
};
use rand::{SeedableRng, rngs::StdRng};

type SphericalProvider = Provider<Spherical, u64>;

fn provider(capacity: usize, rerank: Rerank) -> SphericalProvider {
    let data = Matrix::from_fn(40, 2, |rc| {
        if rc.col == 0 {
            rc.row as f32 + 1.0
        } else {
            0.5
        }
    });
    let quantizer = SphericalQuantizer::train(
        data.as_view(),
        TransformKind::Null,
        SupportedMetric::SquaredL2,
        PreScale::None,
        &mut StdRng::seed_from_u64(42),
        GlobalAllocator,
    )
    .unwrap()
    .as_quantizer::<4>()
    .unwrap();
    Provider::new(
        Spherical::config(
            quantizer,
            Capacity::new(capacity),
            MaxDegree::new(8),
            Matrix::row_vector(vec![0.0, 0.0].into()),
            rerank,
        )
        .unwrap(),
    )
    .unwrap()
}

fn save(provider: &mut SphericalProvider) -> Vec<u8> {
    let mut bytes = Vec::new();
    provider.save(&mut bytes).unwrap();
    bytes
}

#[tokio::test]
async fn search_and_updates_after_restore() {
    for rerank in [Rerank::None, Rerank::F16, Rerank::F32] {
        for capacity in [40, 48] {
            search_and_updates_at_capacity(capacity, rerank).await;
        }
    }
}

async fn search_and_updates_at_capacity(capacity: u32, rerank: Rerank) {
    let provider = provider(40, rerank);
    let config = diskann::graph::config::Builder::new(
        4,
        diskann::graph::config::MaxDegree::new(8),
        20,
        Metric::L2.into(),
    )
    .build()
    .unwrap();
    let mut index = DiskANNIndex::new(config, provider, None);
    for i in 0..35 {
        index
            .insert(
                &Strategy,
                &Context,
                &(i * 10 + 1000),
                &[i as f32 + 1.0, 0.5],
            )
            .await
            .unwrap();
    }
    index.data_provider.delete(&Context, &1070).await.unwrap();
    let knn = Knn::new(10, None).unwrap();
    let mut before = Vec::<Neighbor<u64>>::new();
    index
        .search(knn, &Strategy, &Context, &[12.5, 0.5], &mut before)
        .await
        .unwrap();
    let bytes = save(&mut index.data_provider);
    let restored = SphericalProvider::load_with_options(
        &mut bytes.as_slice(),
        persistence::LoadOptions {
            capacity: Some(capacity),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(restored.len(), 34);
    assert!(!restored.is_empty());
    for i in 0..35 {
        let external = i * 10 + 1000;
        if external == 1070 {
            assert!(restored.to_internal_id(&Context, &external).is_err());
        } else {
            assert_eq!(
                restored.to_internal_id(&Context, &external).unwrap(),
                index
                    .provider()
                    .to_internal_id(&Context, &external)
                    .unwrap()
            );
        }
    }
    drop(index.data_provider);
    let mut restored = DiskANNIndex::new(index.config, restored, None);
    let mut after = Vec::<Neighbor<u64>>::new();
    restored
        .search(knn, &Strategy, &Context, &[12.5, 0.5], &mut after)
        .await
        .unwrap();
    assert_eq!(
        before.iter().map(|n| n.as_tuple()).collect::<Vec<_>>(),
        after.iter().map(|n| n.as_tuple()).collect::<Vec<_>>()
    );
    // Fill the deleted slot, unused saved slots, and all additional capacity.
    for i in 0..u64::from(capacity - 34) {
        restored
            .insert(&Strategy, &Context, &(5000 + i), &[i as f32, 1.5])
            .await
            .unwrap();
    }
    assert!(
        restored
            .insert(&Strategy, &Context, &9999, &[1.0, 2.0])
            .await
            .is_err()
    );
    restored
        .data_provider
        .delete(&Context, &5000)
        .await
        .unwrap();
    restored
        .insert(&Strategy, &Context, &6000, &[0.0, 1.5])
        .await
        .unwrap();
    let bytes = save(&mut restored.data_provider);
    let mut second = SphericalProvider::load(&mut bytes.as_slice()).unwrap();
    assert!(second.to_internal_id(&Context, &5000).is_err());
    assert!(second.to_internal_id(&Context, &6000).is_ok());
    assert_eq!(save(&mut second), bytes);
}

#[tokio::test]
async fn stream_and_file_helpers() {
    for rerank in [Rerank::None, Rerank::F16, Rerank::F32] {
        let mut original = provider(4, rerank);
        original
            .set_element(&Context, &100, &[1.5, 0.5][..])
            .await
            .unwrap();
        let bytes = save(&mut original);
        // Bounded writers fail after the provider header, inside the representation,
        // and near the external-ID trailer, respectively.
        for end in [12, bytes.len() / 2, bytes.len() - 1] {
            let mut output = vec![0; end];
            assert!(original.save(&mut output.as_mut_slice()).is_err());
        }
        let mut stream = bytes.clone();
        stream.extend_from_slice(&bytes);
        stream.extend_from_slice(b"tail");
        let mut cursor = Cursor::new(stream);
        for _ in 0..2 {
            let mut loaded = SphericalProvider::load(&mut cursor).unwrap();
            assert_eq!(save(&mut loaded), bytes);
        }
        assert_eq!(cursor.position(), (2 * bytes.len()) as u64);
        let mut tail = Vec::new();
        cursor.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, b"tail");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot");
        original.save_to_file(&path).unwrap();
        let mut loaded = SphericalProvider::load_from_file(&path).unwrap();
        assert_eq!(save(&mut loaded), bytes);
        original
            .set_element(&Context, &200, &[2.5, 0.5][..])
            .await
            .unwrap();
        original.save_to_file(&path).unwrap();
        let loaded = SphericalProvider::load_from_file_with_options(
            &path,
            persistence::LoadOptions {
                capacity: Some(8),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(loaded.len(), 2);
        for id in 0..6 {
            loaded
                .set_element(&Context, &(300 + id), &[id as f32, 1.5][..])
                .await
                .unwrap();
        }
        assert!(
            loaded
                .set_element(&Context, &999, &[1.0, 1.5][..])
                .await
                .is_err()
        );

        assert!(original.save(&mut FailingWriter).is_err());
        assert!(SphericalProvider::load(&mut FailingReader).is_err());
        // Exercise reader failure inside the representation, beyond the provider header.
        let prefix = Cursor::new(&bytes[..bytes.len() / 2]);
        assert!(SphericalProvider::load(&mut prefix.chain(FailingReader)).is_err());
        for end in [bytes.len() - 1, 12, 20] {
            assert!(SphericalProvider::load(&mut &bytes[..end]).is_err());
        }
    }
}
