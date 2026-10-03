/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Read, Write},
    marker::PhantomData,
    num::{NonZeroU32, NonZeroUsize},
};

use diskann::{ANNError, ANNResult, error::ErrorContext};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use super::{Full, FullPrecision};
use crate::{
    epoch::Registry,
    num::{Bytes, Capacity, MaxDegree},
    persistence::{self, LoadOptions, Snapshot},
    store::{self, Store, intrusive::Intrusive},
};

// Version-1 representation payload:
// * Fixed-size Header (52 bytes), with representation=1 and scalar=1/2/3/4 for
//   f32/f16/u8/i8. Metric codes 0/1/2/3 mean cosine/inner-product/L2/normalized-cosine.
// * capacity+frozen slot records: a boolean followed, when true, by dim scalar values.
//   Every frozen slot must be present. Floating point values retain their exact bits.
// * capacity+frozen adjacency lists: a u32 length followed by that many u32 IDs.
// All numeric fields are little-endian. Any incompatible payload change needs a new
// provider snapshot version. Do not serialize Rust layouts or enum discriminants.
#[derive(Debug, Serialize, Deserialize)]
struct Header {
    representation: u32,
    scalar: u32,
    capacity: u32,
    max_degree: u32,
    frozen: u32,
    dim: u64,
    metric: i32,
    lookahead: u64,
    epoch_guard_slots: u64,
    freelist_recycle_capacity: u32,
}

impl<T: FullPrecision> Full<T> {
    fn save_full<W: Write>(&mut self, writer: &mut W, scalar: u32) -> ANNResult<()> {
        let (epoch_guard_slots, freelist_recycle_capacity) =
            self.store.config().snapshot_parameters();
        let header = Header {
            representation: 1,
            scalar,
            capacity: self.store.capacity().value() as u32,
            max_degree: self.store.neighbors().max_degree_u32(),
            frozen: self.store.frozen().len() as u32,
            dim: self.dim() as u64,
            metric: match self.metric {
                Metric::Cosine => 0,
                Metric::InnerProduct => 1,
                Metric::L2 => 2,
                Metric::CosineNormalized => 3,
            },
            lookahead: self.lookahead.map_or(0, |v| v.get() as u64),
            epoch_guard_slots: epoch_guard_slots.get() as u64,
            freelist_recycle_capacity: freelist_recycle_capacity.get(),
        };
        persistence::write(writer, &header)?;
        let reader = self.reader()?;
        for id in 0..self.store.id_limit().value() {
            let vector = reader.read(id as usize);
            if id >= header.capacity && vector.is_none() {
                return Err(ANNError::message("snapshot frozen vector is missing"));
            }
            persistence::write(writer, &vector.is_some())?;
            if let Some(vector) = vector {
                write_vector::<T, _>(writer, vector)?;
            }
        }
        persistence::save_graph(self.store.neighbors(), writer)
    }

    fn load_full<R: Read>(reader: &mut R, scalar: u32, options: LoadOptions) -> ANNResult<Self> {
        let header: Header = persistence::read(reader)?;
        if header.representation != 1 || header.scalar != scalar {
            return Err(ANNError::message(
                "snapshot representation or scalar type mismatch",
            ));
        }
        let metric = match header.metric {
            0 => Metric::Cosine,
            1 => Metric::InnerProduct,
            2 => Metric::L2,
            3 => Metric::CosineNormalized,
            _ => return Err(ANNError::message("unsupported snapshot distance metric")),
        };
        let dim = usize::try_from(header.dim)?;
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
        let overflow = || ANNError::message("snapshot vector or graph allocation overflows");
        let bytes = dim.checked_mul(size_of::<T>()).ok_or_else(overflow)?;

        // Check both large store allocations before Store::new allocates even its tags
        // or registry. Include intrusive tags, cache-line padding, and graph lengths.
        let stride = bytes
            .checked_add(1)
            .and_then(|n| n.checked_next_multiple_of(Bytes::CACHELINE.value()))
            .ok_or_else(overflow)?;
        let graph_stride = (header.max_degree as usize)
            .checked_add(1)
            .and_then(|n| n.checked_mul(size_of::<u32>()))
            .ok_or_else(overflow)?;
        for stride in [stride, graph_stride] {
            let allocation = stride.checked_mul(entries as usize).ok_or_else(overflow)?;
            std::alloc::Layout::from_size_align(allocation, 128)?;
        }
        Registry::validate_capacity(epoch_guard_slots)?;

        let mut config = store::Config::new();
        config
            .epoch_guard_slots(epoch_guard_slots)
            .freelist_recycle_capacity(freelist_recycle_capacity);
        let layout = store::Layout::new(
            Capacity::new(capacity as usize),
            MaxDegree::new(header.max_degree as usize),
            header.frozen,
        );
        let mut store = Store::new(layout, config, Intrusive::config(Bytes::new(bytes)))?;
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
            let data = slot.data().as_mut_slice();
            reader.read_exact(data).context("reading snapshot vector")?;
            if cfg!(target_endian = "big") {
                swap_element_bytes(data, size_of::<T>());
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
            metric,
            lookahead,
            _type: PhantomData,
        })
    }
}

fn write_vector<T, W: Write>(writer: &mut W, vector: &[u8]) -> ANNResult<()> {
    if cfg!(target_endian = "little") {
        writer.write_all(vector)?;
    } else {
        // Fixed-size scratch space, regardless of vector dimension. The chunk size is
        // divisible by every supported scalar width, so elements never straddle chunks.
        let mut scratch = [0; 8192];
        for chunk in vector.chunks(scratch.len()) {
            let output = &mut scratch[..chunk.len()];
            output.copy_from_slice(chunk);
            swap_element_bytes(output, size_of::<T>());
            writer.write_all(output)?;
        }
    }
    Ok(())
}

fn swap_element_bytes(data: &mut [u8], element_bytes: usize) {
    for element in data.chunks_exact_mut(element_bytes) {
        element.reverse();
    }
}

macro_rules! impl_snapshot {
    ($ty:ty, $code:literal) => {
        impl Snapshot for Full<$ty> {
            fn write_snapshot<W: Write>(&mut self, writer: &mut W) -> ANNResult<()> {
                self.save_full(writer, $code)
            }

            fn read_snapshot<R: Read>(reader: &mut R) -> ANNResult<Self> {
                Self::load_full(reader, $code, LoadOptions::default())
            }
            fn read_snapshot_with_options<R: Read>(
                reader: &mut R,
                options: LoadOptions,
            ) -> ANNResult<Self> {
                Self::load_full(reader, $code, options)
            }
        }
    };
}

impl_snapshot!(f32, 1);
impl_snapshot!(half::f16, 2);
impl_snapshot!(u8, 3);
impl_snapshot!(i8, 4);

#[cfg(test)]
mod tests;
