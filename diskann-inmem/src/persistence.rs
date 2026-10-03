/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Snapshots of in-memory providers.
//!
//! [`crate::Provider::save`] requires exclusive access, so all searches and updates must
//! finish before saving. An index held in an `Arc` must first regain exclusive ownership.
//! Snapshots preserve capacity, configuration, internal IDs, graph edges, external IDs,
//! live vectors, and frozen start points. Deleted slots become reusable on load; locks,
//! epoch state, and event counters are recreated.
//!
//! ```
//! use diskann_inmem::{Provider, repr::Full, num::{Capacity, MaxDegree}};
//! use diskann_utils::views::Matrix;
//! use diskann_vector::distance::Metric;
//!
//! # fn main() -> diskann::ANNResult<()> {
//! let config = Full::config(
//!     Capacity::new(100), MaxDegree::new(16), Metric::L2,
//!     Matrix::row_vector(vec![0.0f32, 0.0].into()),
//! )?;
//! let mut provider = Provider::<_, String>::new(config)?;
//! let mut bytes = Vec::new();
//! provider.save(&mut bytes)?;
//! let restored = Provider::<Full<f32>, String>::load(&mut bytes.as_slice())?;
//! # Ok(())
//! # }
//! ```
//!
//! For a [`diskann::graph::DiskANNIndex`], save through `index.data_provider.save(...)`.
//! Reconstruct the index with `DiskANNIndex::new(graph_config, restored, thread_hint)`;
//! algorithm parameters such as construction alpha and search-list size belong to the
//! caller and are not part of the provider snapshot.
//!
//! # Format and compatibility
//!
//! Version 1 starts with `DANNIMEM` and a little-endian `u32` version. The representation
//! writes its own tagged payload, followed by a `u32` mapping count and that many records:
//! an internal `u32` ID, a `u64` byte length, and the serialized external ID. All integers
//! use fixed-width little-endian encoding. External IDs use bincode 1 with these settings;
//! loading requires the same external-ID type and serde schema. There is no automatic
//! schema migration or compatibility with the older DiskANN index files.
//!
//! Only [`crate::repr::Full`] over `f32`, `half::f16`, `u8`, and `i8` implements [`Snapshot`].
//! Representation and scalar mismatches are rejected. Loads validate structure and
//! allocation arithmetic but still require enough memory for the saved capacity.
//!
//! Stream methods consume/write one snapshot without closing or flushing the stream.
//! Callers handle partial output on failure. File helpers use buffered I/O and replace
//! the destination only after a successful write, flush, and file sync. Atomic replacement
//! does not promise directory-entry durability across a power failure.

use std::io::{Read, Write};

use bincode::Options;
use diskann::{ANNError, ANNResult, graph::AdjacencyList};
use serde::{Serialize, de::DeserializeOwned};

use crate::{neighbors::Neighbors, repr::Representation};

/// Opt-in persistence for a provider's data representation.
///
/// Implementations must tag and validate their payload, preserve internal IDs, capacity,
/// configuration, frozen points and graph edges, and restore a store ready for updates.
/// They must validate all graph IDs before installing adjacency lists. Do not persist
/// raw synchronization objects, pointers, padding, or unpublished vector bytes.
pub trait Snapshot: Representation + Sized {
    /// Write a representation payload while holding exclusive access.
    fn write_snapshot<W: Write>(&mut self, writer: &mut W) -> ANNResult<()>;

    /// Read exactly one representation payload, rejecting incompatible formats.
    fn read_snapshot<R: Read>(reader: &mut R) -> ANNResult<Self>;
}

// TODO(quantization): Implement Snapshot for repr::Spherical, persisting the trained
// quantizer via diskann-quantization/flatbuffers, compressed vectors (including frozen
// points), and optional f16 reranking vectors/configuration. Restore those bytes directly
// without retraining or recompressing, and validate the compressed layout and bit width.

pub(crate) const MAGIC: &[u8; 8] = b"DANNIMEM";
pub(crate) const VERSION: u32 = 1;

pub(crate) fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
}

pub(crate) fn write<W: Write, T: Serialize>(writer: &mut W, value: &T) -> ANNResult<()> {
    codec().serialize_into(writer, value).map_err(ANNError::new)
}

pub(crate) fn read<R: Read, T: DeserializeOwned>(reader: &mut R) -> ANNResult<T> {
    codec().deserialize_from(reader).map_err(ANNError::new)
}

pub(crate) fn save_graph<W: Write>(graph: &Neighbors, writer: &mut W) -> ANNResult<()> {
    let mut neighbors = AdjacencyList::new();
    for id in 0..graph.entries() {
        graph.get(id, &mut neighbors)?;
        if neighbors.iter().any(|&n| n >= graph.entries()) {
            return Err(ANNError::message(
                "snapshot graph contains an out-of-range neighbor",
            ));
        }
        write(writer, &(neighbors.len() as u32))?;
        for neighbor in neighbors.iter() {
            write(writer, neighbor)?;
        }
    }
    Ok(())
}

pub(crate) fn load_graph<R: Read>(graph: &Neighbors, reader: &mut R) -> ANNResult<()> {
    let mut neighbors = Vec::new();
    for id in 0..graph.entries() {
        let len: u32 = read(reader)?;
        if len > graph.max_degree_u32() {
            return Err(ANNError::message(
                "snapshot adjacency list exceeds maximum degree",
            ));
        }
        neighbors.clear();
        for _ in 0..len {
            let neighbor: u32 = read(reader)?;
            if neighbor >= graph.entries() {
                return Err(ANNError::message("snapshot neighbor is out of range"));
            }
            neighbors.push(neighbor);
        }
        // Deleted slots may still be referenced, and order must be preserved.
        graph.set(id, &neighbors)?;
    }
    Ok(())
}
