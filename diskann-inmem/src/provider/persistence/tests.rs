/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{self, Cursor};

use diskann::{
    graph::{DiskANNIndex, InplaceDeleteMethod, search::Knn},
    neighbor::Neighbor,
    provider::{DataProvider, Delete, SetElement},
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;
use serde::Deserialize;

use super::*;
use crate::{
    Context, Strategy,
    num::{Capacity, MaxDegree},
    repr::{Full, Representation},
};

type TestProvider = Provider<Full<f32>, u64>;

fn provider<M: Id>(capacity: usize) -> Provider<Full<f32>, M> {
    Provider::new(
        Full::config(
            Capacity::new(capacity),
            MaxDegree::new(8),
            Metric::L2,
            Matrix::row_vector(vec![0f32, 0.0].into()),
        )
        .unwrap(),
    )
    .unwrap()
}

fn save<M: Id + Serialize>(provider: &mut Provider<Full<f32>, M>) -> Vec<u8> {
    let mut bytes = Vec::new();
    provider.save(&mut bytes).unwrap();
    bytes
}

#[tokio::test]
async fn search_and_updates_after_restore() {
    let provider = provider::<u64>(40);
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
    index
        .inplace_delete(
            Strategy,
            &Context,
            &1070,
            3,
            InplaceDeleteMethod::VisitedAndTopK {
                k_value: 10,
                l_value: 20,
            },
        )
        .await
        .unwrap();
    let knn = Knn::new(10, None).unwrap();
    let mut before = Vec::<Neighbor<u64>>::new();
    index
        .search(knn, &Strategy, &Context, &[12.5, 0.5], &mut before)
        .await
        .unwrap();
    let bytes = save(&mut index.data_provider);
    let restored = TestProvider::load(&mut bytes.as_slice()).unwrap();
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
    // Fill all six free slots, including the deleted one.
    for i in 0..6 {
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
        .inplace_delete(
            Strategy,
            &Context,
            &5000,
            3,
            InplaceDeleteMethod::VisitedAndTopK {
                k_value: 10,
                l_value: 20,
            },
        )
        .await
        .unwrap();
    restored
        .insert(&Strategy, &Context, &6000, &[0.0, 1.5])
        .await
        .unwrap();
    let bytes = save(&mut restored.data_provider);
    let mut second = TestProvider::load(&mut bytes.as_slice()).unwrap();
    assert!(second.to_internal_id(&Context, &5000).is_err());
    assert!(second.to_internal_id(&Context, &6000).is_ok());
    assert_eq!(save(&mut second), bytes);
}

#[tokio::test]
async fn sparse_and_full_snapshots_reuse_free_slots() {
    for count in [0, 35, 40] {
        let mut original = provider::<u64>(40);
        for id in 0..count {
            original
                .set_element(&Context, &id, &[id as f32, 0.0][..])
                .await
                .unwrap();
        }
        let bytes = save(&mut original);
        let mut loaded = TestProvider::load(&mut bytes.as_slice()).unwrap();
        // In the 35-point case, minting from zero would hit 20 occupied slots and fail.
        for id in count..40 {
            loaded
                .set_element(&Context, &id, &[id as f32, 0.0][..])
                .await
                .unwrap();
        }
        assert!(
            loaded
                .set_element(&Context, &40, &[1.0, 2.0][..])
                .await
                .is_err()
        );
        for id in 0..40 {
            loaded.delete(&Context, &id).await.unwrap();
        }
        let bytes = save(&mut loaded);
        let loaded = TestProvider::load(&mut bytes.as_slice()).unwrap();
        for id in 0..40 {
            assert!(loaded.to_internal_id(&Context, &id).is_err());
            loaded
                .set_element(&Context, &(id + 100), &[0.0, 0.0][..])
                .await
                .unwrap();
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct CustomId {
    tenant: String,
    key: u64,
}

async fn id_roundtrip<M: Id + Serialize + DeserializeOwned + std::fmt::Debug>(ids: &[M]) {
    let mut original = provider(ids.len());
    for id in ids {
        original
            .set_element(&Context, id, &[1.0, 2.0][..])
            .await
            .unwrap();
    }
    let bytes = save(&mut original);
    let loaded = Provider::<Full<f32>, M>::load(&mut bytes.as_slice()).unwrap();
    for id in ids {
        let internal = original.to_internal_id(&Context, id).unwrap();
        assert_eq!(loaded.to_internal_id(&Context, id).unwrap(), internal);
        assert_eq!(&loaded.to_external_id(&Context, internal).unwrap(), id);
    }
}

#[tokio::test]
async fn external_id_schemas() {
    id_roundtrip(&[0u64, u64::MAX, 123456]).await;
    id_roundtrip(&["".to_owned(), "vector/你好".to_owned()]).await;
    id_roundtrip(&[
        CustomId {
            tenant: "alpha".into(),
            key: 3,
        },
        CustomId {
            tenant: "beta".into(),
            key: 3,
        },
    ])
    .await;
}

#[tokio::test]
async fn reject_invalid_mappings_and_truncation() {
    let mut original = provider::<u64>(3);
    for id in [100, 200] {
        original
            .set_element(&Context, &id, &[1.0, 2.0][..])
            .await
            .unwrap();
    }
    let bytes = save(&mut original);
    for end in 0..bytes.len() {
        assert!(
            TestProvider::load(&mut &bytes[..end]).is_err(),
            "offset {end}"
        );
    }
    let mut cursor = Cursor::new(&bytes[12..]);
    let _ = Full::<f32>::read_snapshot(&mut cursor).unwrap();
    let mapping = 12 + cursor.position() as usize;
    let first = mapping + 4;
    let second = first + 4 + 8 + 8;
    for (offset, replacement) in [
        (0, b"NOTMAGIC".to_vec()),
        (8, 99u32.to_le_bytes().to_vec()),
        (mapping, 4u32.to_le_bytes().to_vec()),
        (first, 2u32.to_le_bytes().to_vec()), // Vacant slot.
        (first, 3u32.to_le_bytes().to_vec()), // Frozen slot.
        (first, u32::MAX.to_le_bytes().to_vec()),
        (second, 0u32.to_le_bytes().to_vec()), // Duplicate internal ID.
        (second + 12, 100u64.to_le_bytes().to_vec()), // Duplicate external ID.
        (first + 4, 7u64.to_le_bytes().to_vec()), // ID length too short.
        (first + 4, 9u64.to_le_bytes().to_vec()), // ID length too long.
    ] {
        let mut changed = bytes.clone();
        changed[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert!(
            TestProvider::load(&mut changed.as_slice()).is_err(),
            "offset {offset}"
        );
    }
    let internal = original.to_internal_id(&Context, &100).unwrap();
    original.representation.retire(internal).unwrap();
    assert!(original.save(&mut Vec::new()).is_err());
}

#[test]
fn stream_consumes_one_snapshot() {
    let mut original = provider::<u64>(0);
    let mut bytes = save(&mut original);
    let end = bytes.len();
    bytes.extend_from_slice(b"application data");
    let mut cursor = Cursor::new(bytes);
    TestProvider::load(&mut cursor).unwrap();
    assert_eq!(cursor.position() as usize, end);
}

struct FailingWriter;
impl Write for FailingWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("write failed"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct FailingReader;
impl Read for FailingReader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("read failed"))
    }
}

#[test]
fn io_errors_are_returned() {
    assert!(provider::<u64>(1).save(&mut FailingWriter).is_err());
    assert!(TestProvider::load(&mut FailingReader).is_err());
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FailingId;
impl Serialize for FailingId {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("serialization failed"))
    }
}

#[tokio::test]
async fn files_replace_successfully_and_preserve_previous_on_failure() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.snapshot");
    let mut original = provider::<u64>(3);
    original.save_to_file(&path).unwrap();
    original
        .set_element(&Context, &99, &[1.0, 2.0][..])
        .await
        .unwrap();
    original.save_to_file(&path).unwrap();
    let loaded = TestProvider::load_from_file(&path).unwrap();
    assert_eq!(loaded.to_internal_id(&Context, &99).unwrap(), 0);
    let expected = std::fs::read(&path).unwrap();
    let mut bad = provider::<FailingId>(1);
    bad.set_element(&Context, &FailingId, &[1.0, 2.0][..])
        .await
        .unwrap();
    assert!(bad.save_to_file(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
    assert!(
        original
            .save_to_file(temp.path().join("missing/snapshot"))
            .is_err()
    );
    assert!(TestProvider::load_from_file(temp.path().join("missing")).is_err());
}
