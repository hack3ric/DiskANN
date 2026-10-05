/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Read, Write},
    num::{NonZeroU32, NonZeroUsize},
};

use diskann::{ANNError, ANNResult, error::ErrorContext};
use diskann_quantization::{
    alloc::GlobalAllocator,
    spherical::{DataMeta, iface},
};
use serde::{Deserialize, Serialize};

use super::{Rerank, Reranker, Spherical};
use crate::{
    epoch::Registry,
    num::{Bytes, Capacity, MaxDegree},
    persistence::{self, LoadOptions, Snapshot},
    store::{self, Store, cons, intrusive::Intrusive},
};

// Provider version 1, representation 2. All fields use fixed-width little-endian
// encoding. Layout 1 is dense packed codes followed by DataMeta's three u16 words
// (two f16 bit patterns and bit_sum). No store tags or alignment padding are saved.
// Header, quantizer blob, capacity+frozen slot records, then graph adjacency lists.
// Each slot is a bool followed, if present, by compressed bytes and optional reranking
// values: none (tag 0), dim f16s (tag 1), or dim f32s (tag 2).
// A change to the packed representation requires a new layout version.
#[derive(Debug, Serialize, Deserialize)]
struct Header {
    representation: u32,
    layout: u32,
    capacity: u32,
    max_degree: u32,
    frozen: u32,
    full_dim: u64,
    dim: u64,
    nbits: u32,
    bytes: u64,
    rerank: u32,
    lookahead: u64,
    epoch_guard_slots: u64,
    freelist_recycle_capacity: u32,
    quantizer_len: u64,
}

impl Snapshot for Spherical {
    fn write_snapshot<W: Write>(&mut self, writer: &mut W) -> ANNResult<()> {
        let quantizer = self
            .quantizer
            .serialize(GlobalAllocator)
            .map_err(ANNError::new)?;
        let (epoch_guard_slots, freelist_recycle_capacity) =
            self.store.config().snapshot_parameters();
        let header = Header {
            representation: 2,
            layout: 1,
            capacity: self.store.capacity().value() as u32,
            max_degree: self.store.neighbors().max_degree_u32(),
            frozen: self.store.frozen().len() as u32,
            full_dim: self.full_dim as u64,
            dim: self.quantizer.dim() as u64,
            nbits: self.quantizer.nbits() as u32,
            bytes: self.quantizer.bytes() as u64,
            rerank: match self.reranker {
                Reranker::None => 0,
                Reranker::F16(_) => 1,
                Reranker::F32(_) => 2,
            },
            lookahead: self.lookahead.map_or(0, |v| v.get() as u64),
            epoch_guard_slots: epoch_guard_slots.get() as u64,
            freelist_recycle_capacity: freelist_recycle_capacity.get(),
            quantizer_len: quantizer.len() as u64,
        };
        persistence::write(writer, &header)?;
        writer.write_all(&quantizer)?;
        self.store.guard(|slots, guard| -> ANNResult<()> {
            let compressed = slots.first().reader(guard.share());
            let rerank = slots.second().slots().map(|s| s.reader(guard));
            for id in 0..self.store.id_limit().value() {
                let vector = compressed.read(id as usize);
                if id >= header.capacity && vector.is_none() {
                    return Err(ANNError::message("snapshot frozen vector is missing"));
                }
                persistence::write(writer, &vector.is_some())?;
                if let Some(vector) = vector {
                    let (codes, meta) = vector.split_at(vector.len() - size_of::<DataMeta>());
                    writer.write_all(codes)?;
                    write_words::<2, _>(writer, meta)?;
                    if let Some(rerank) = &rerank {
                        let vector = rerank.read(id as usize).ok_or_else(|| {
                            ANNError::message("snapshot reranking vector is missing")
                        })?;
                        match self.reranker {
                            Reranker::F16(_) => write_words::<2, _>(writer, vector)?,
                            Reranker::F32(_) => write_words::<4, _>(writer, vector)?,
                            Reranker::None => unreachable!("reranking store without reranker"),
                        }
                    }
                }
            }
            Ok(())
        })??;
        persistence::save_graph(self.store.neighbors(), writer)
    }

    fn read_snapshot<R: Read>(reader: &mut R) -> ANNResult<Self> {
        Self::read_snapshot_with_options(reader, LoadOptions::default())
    }

    fn read_snapshot_with_options<R: Read>(
        reader: &mut R,
        options: LoadOptions,
    ) -> ANNResult<Self> {
        let header: Header = persistence::read(reader)?;
        if header.representation != 2 || header.layout != 1 {
            return Err(ANNError::message(
                "snapshot representation or spherical layout mismatch",
            ));
        }
        let rerank = match header.rerank {
            0 => Rerank::None,
            1 => Rerank::F16,
            2 => Rerank::F32,
            _ => return Err(ANNError::message("unsupported snapshot reranking mode")),
        };
        if !matches!(header.nbits, 1 | 2 | 4 | 8) || header.frozen == 0 {
            return Err(ANNError::message(
                "invalid spherical snapshot bit width or frozen count",
            ));
        }
        let full_dim = usize::try_from(header.full_dim)?;
        let dim = usize::try_from(header.dim)?;
        let overflow = || ANNError::message("snapshot vector or graph allocation overflows");
        let bytes = dim
            .checked_mul(header.nbits as usize)
            .and_then(|n| n.checked_add(7))
            .map(|n| n / 8)
            .and_then(|n| n.checked_add(size_of::<DataMeta>()))
            .ok_or_else(overflow)?;
        if header.bytes != bytes as u64 {
            return Err(ANNError::message(
                "snapshot compressed vector size mismatch",
            ));
        }
        let lookahead = options
            .prefetch
            .unwrap_or(NonZeroUsize::new(usize::try_from(header.lookahead)?));
        let epoch_guard_slots = NonZeroUsize::new(usize::try_from(header.epoch_guard_slots)?)
            .ok_or_else(|| ANNError::message("snapshot epoch guard slots must be nonzero"))?;
        let freelist_recycle_capacity = NonZeroU32::new(header.freelist_recycle_capacity)
            .ok_or_else(|| ANNError::message("snapshot freelist capacity must be nonzero"))?;
        let entries = header
            .capacity
            .checked_add(header.frozen)
            .ok_or_else(|| ANNError::message("snapshot slot count overflows u32"))?;
        let saved_entries = entries;
        let capacity = options.capacity.unwrap_or(header.capacity);
        if capacity < header.capacity {
            return Err(ANNError::message("snapshot capacity cannot shrink"));
        }
        let entries = capacity
            .checked_add(header.frozen)
            .ok_or_else(|| ANNError::message("snapshot slot count overflows u32"))?;
        let epoch_guard_slots = options.epoch_guard_slots.unwrap_or(epoch_guard_slots);
        let remap = |id| {
            if id >= header.capacity {
                id + (capacity - header.capacity)
            } else {
                id
            }
        };
        let compressed_stride = bytes
            .checked_add(1)
            .and_then(|n| n.checked_next_multiple_of(Bytes::CACHELINE.value()))
            .ok_or_else(overflow)?;
        let rerank_element_bytes = match rerank {
            Rerank::None => 0,
            Rerank::F16 => size_of::<half::f16>(),
            Rerank::F32 => size_of::<f32>(),
        };
        let rerank_stride = full_dim
            .checked_mul(rerank_element_bytes)
            .and_then(|n| n.checked_next_multiple_of(Bytes::CACHELINE.value()))
            .ok_or_else(overflow)?;
        let graph_stride = (header.max_degree as usize)
            .checked_add(1)
            .and_then(|n| n.checked_mul(size_of::<u32>()))
            .ok_or_else(overflow)?;
        for stride in [compressed_stride, rerank_stride, graph_stride] {
            let allocation = stride.checked_mul(entries as usize).ok_or_else(overflow)?;
            std::alloc::Layout::from_size_align(allocation, 128)?;
        }
        Registry::validate_capacity(epoch_guard_slots)?;

        // FlatBuffers identifiers require at least eight bytes. Read incrementally so
        // a truncated blob with a large declared length does not allocate that length.
        if !(8..(i32::MAX as u64)).contains(&header.quantizer_len) {
            return Err(ANNError::message("invalid snapshot quantizer length"));
        }
        let mut blob = Vec::new();
        reader.take(header.quantizer_len).read_to_end(&mut blob)?;
        if blob.len() as u64 != header.quantizer_len {
            return Err(ANNError::message("truncated snapshot quantizer"));
        }
        let quantizer = iface::try_deserialize::<GlobalAllocator, _>(&blob, GlobalAllocator)
            .map_err(ANNError::new)
            .context("reading snapshot quantizer")?;
        if quantizer.full_dim() != full_dim
            || quantizer.dim() != dim
            || quantizer.nbits() != header.nbits as usize
            || quantizer.bytes() != bytes
        {
            return Err(ANNError::message(
                "snapshot quantizer and compressed layout disagree",
            ));
        }
        let (reranker, rerank_config) =
            Reranker::new_with_config(rerank, quantizer.metric(), full_dim);
        let mut config = store::Config::new();
        config
            .epoch_guard_slots(epoch_guard_slots)
            .freelist_recycle_capacity(freelist_recycle_capacity);
        let layout = store::Layout::new(
            Capacity::new(capacity as usize),
            MaxDegree::new(header.max_degree as usize),
            header.frozen,
        );
        let mut store = Store::new(
            layout,
            config,
            cons::Config::new(Intrusive::config(Bytes::new(bytes)), rerank_config),
        )?;
        for id in 0..saved_entries {
            let present: bool = persistence::read(reader)?;
            if !present {
                if id >= header.capacity {
                    return Err(ANNError::message("snapshot frozen vector is missing"));
                }
                continue;
            }
            let mut slot = store
                .slot(remap(id))
                .ok_or_else(|| ANNError::message("could not restore snapshot slot"))?;
            let data = slot.data().first().as_mut_slice();
            reader
                .read_exact(data)
                .context("reading snapshot compressed vector")?;
            native_words::<2>(&mut data[bytes - size_of::<DataMeta>()..]);
            if let Some(rerank) = slot.data().second() {
                let data = rerank.as_mut_slice();
                reader
                    .read_exact(data)
                    .context("reading snapshot reranking vector")?;
                match reranker {
                    Reranker::F16(_) => native_words::<2>(data),
                    Reranker::F32(_) => native_words::<4>(data),
                    Reranker::None => unreachable!("reranking store without reranker"),
                }
            }
            if id >= header.capacity {
                slot.freeze();
            } else {
                slot.publish();
            }
        }
        persistence::load_graph(store.neighbors(), reader, header.capacity, saved_entries)?;
        store.finish_restore();
        Ok(Self {
            store,
            quantizer,
            full_dim,
            lookahead,
            reranker,
        })
    }
}

// Metadata and f16 reranking use 2-byte words; f32 reranking uses 4-byte words.
// Preserve all float bits, including signed zero, without floating point arithmetic.
fn write_words<const N: usize, W: Write>(writer: &mut W, data: &[u8]) -> ANNResult<()> {
    if cfg!(target_endian = "little") {
        writer.write_all(data)?;
    } else {
        for word in data.as_chunks::<N>().0 {
            let mut word = *word;
            word.reverse();
            writer.write_all(&word)?;
        }
    }
    Ok(())
}

fn native_words<const N: usize>(data: &mut [u8]) {
    if cfg!(target_endian = "big") {
        for word in data.as_chunks_mut::<N>().0 {
            word.reverse();
        }
    }
}

#[cfg(test)]
mod tests;
