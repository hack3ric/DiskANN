/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use bincode::Options;
use diskann::{ANNError, ANNResult, error::ErrorContext};
use serde::{Serialize, de::DeserializeOwned};

use super::{Id, Provider};
use crate::{
    counters::Counters,
    ids::IdMap,
    persistence::{self, Snapshot},
};

impl<R: Snapshot, M: Id> Provider<R, M> {
    /// Save one snapshot, preserving the representation and external ID mappings.
    ///
    /// All searches and updates must finish before obtaining `&mut self`. The writer is
    /// not flushed; a failure can leave partial output. See [`crate::persistence`].
    ///
    /// # Errors
    /// Returns an error on inconsistent provider state, serialization or I/O failure.
    pub fn save<W: Write>(&mut self, writer: &mut W) -> ANNResult<()>
    where
        M: Serialize,
    {
        let count = self.validate_snapshot_mapping()?;
        writer.write_all(persistence::MAGIC)?;
        persistence::write(writer, &persistence::VERSION)?;
        self.representation
            .write_snapshot(writer)
            .context("saving snapshot representation")?;
        persistence::write(writer, &count)?;
        for id in 0..u32::try_from(self.representation.capacity().value())? {
            if let Some(external) = self.mapping.to_external(id) {
                // Buffer only a single ID to know its length before writing it.
                let bytes = persistence::codec()
                    .serialize(&external)
                    .map_err(ANNError::new)
                    .context("serializing snapshot external ID")?;
                persistence::write(writer, &id)?;
                persistence::write(writer, &(bytes.len() as u64))?;
                writer.write_all(&bytes)?;
            }
        }
        Ok(())
    }

    /// Load one snapshot with its saved capacity and tuning parameters.
    ///
    /// Use the same representation, scalar type, and external-ID serde schema that were
    /// used when saving. The returned provider is ready for searches and updates. The
    /// reader is left immediately after this snapshot; trailing application data is allowed.
    ///
    /// # Errors
    /// Rejects incompatible versions/types, malformed data, and inconsistent ID mappings.
    /// I/O and deserialization errors are propagated without exposing partial state.
    pub fn load<S: Read>(reader: &mut S) -> ANNResult<Self>
    where
        M: DeserializeOwned,
    {
        Self::load_with_options(reader, persistence::LoadOptions::default())
    }

    /// Load a snapshot with optional capacity growth and runtime tuning overrides.
    pub fn load_with_options<S: Read>(
        reader: &mut S,
        options: persistence::LoadOptions,
    ) -> ANNResult<Self>
    where
        M: DeserializeOwned,
    {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != persistence::MAGIC {
            return Err(ANNError::message("invalid provider snapshot magic"));
        }
        let version: u32 = persistence::read(reader)?;
        if version != persistence::VERSION {
            return Err(ANNError::message(format!(
                "unsupported provider snapshot version {version}"
            )));
        }
        let representation = R::read_snapshot_with_options(reader, options)
            .context("loading snapshot representation")?;
        let capacity = u32::try_from(representation.capacity().value())?;
        let count: u32 = persistence::read(reader)?;
        let live = (0..capacity)
            .filter(|&id| representation.is_readable(id) == Some(true))
            .count();
        if count as usize != live {
            return Err(ANNError::message(
                "snapshot mapping count does not match live slots",
            ));
        }
        let mapping = IdMap::new(representation.capacity());
        for _ in 0..count {
            let id: u32 = persistence::read(reader)?;
            if id >= capacity || representation.is_readable(id) != Some(true) {
                return Err(ANNError::message(
                    "snapshot mapping references a non-live writable slot",
                ));
            }
            let len: u64 = persistence::read(reader)?;
            let mut limited = reader.take(len);
            let external: M = persistence::codec()
                .with_limit(len)
                .deserialize_from(&mut limited)
                .map_err(ANNError::new)
                .context("deserializing snapshot external ID")?;
            if limited.limit() != 0 {
                return Err(ANNError::message("snapshot external ID has trailing bytes"));
            }
            mapping
                .insert(external, id)
                .context("restoring snapshot ID mapping")?;
        }
        Ok(Self {
            representation,
            mapping,
            counters: Counters::new(),
        })
    }

    /// Save to a temporary file beside `path`, then atomically replace `path`.
    ///
    /// The file is flushed and synced before replacement. The parent directory must
    /// already exist. A failure before replacement leaves an existing snapshot intact.
    pub fn save_to_file(&mut self, path: impl AsRef<Path>) -> ANNResult<()>
    where
        M: Serialize,
    {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut temp = tempfile::NamedTempFile::new_in(parent)?;
        {
            let mut writer = BufWriter::new(temp.as_file_mut());
            self.save(&mut writer)?;
            writer.flush()?;
        }
        temp.as_file().sync_all()?;
        temp.persist(path)
            .map_err(|error| ANNError::new(error.error))?;
        Ok(())
    }

    /// Load a snapshot from a file using buffered I/O. See [`Self::load`].
    pub fn load_from_file(path: impl AsRef<Path>) -> ANNResult<Self>
    where
        M: DeserializeOwned,
    {
        Self::load(&mut BufReader::new(File::open(path)?))
    }

    /// Load a snapshot file with runtime overrides. See [`Self::load_with_options`].
    pub fn load_from_file_with_options(
        path: impl AsRef<Path>,
        options: persistence::LoadOptions,
    ) -> ANNResult<Self>
    where
        M: DeserializeOwned,
    {
        Self::load_with_options(&mut BufReader::new(File::open(path)?), options)
    }

    fn validate_snapshot_mapping(&self) -> ANNResult<u32> {
        let mut count = 0;
        for id in 0..u32::try_from(self.representation.capacity().value())? {
            let external = self.mapping.to_external(id);
            if self.representation.is_readable(id) != Some(external.is_some()) {
                return Err(ANNError::message(
                    "snapshot ID mapping and slot occupancy disagree",
                ));
            }
            if let Some(external) = external {
                if self.mapping.to_internal(&external) != Some(id) {
                    return Err(ANNError::message("snapshot ID mapping is inconsistent"));
                }
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
