/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Opt-in, uniform placement of live vector-storage base pages under one DRAM budget.
//! Policy and reports are runtime-only; snapshots never serialize placement state.
//! Maintenance requires quiescence, supplied by the caller between workload batches.

use std::{collections::BinaryHeap, io, time::Instant};

use crate::buffer::Buffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorPlacementOptions {
    pub budget_bytes: u64,
    pub dram_node: u32,
    pub cxl_node: u32,
    pub seed: u64,
}

impl Default for VectorPlacementOptions {
    fn default() -> Self {
        Self {
            budget_bytes: 0,
            dram_node: 0,
            cxl_node: 1,
            seed: 0,
        }
    }
}

/// Targets describe managed storage, not process RSS. Migration counters count pages
/// successfully submitted/confirmed on the destination, including no-ops.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PlacementReport {
    pub page_bytes: u64,
    /// Sum of readable row strides, including padding and inline metadata.
    pub active_bytes: u64,
    /// Unique base pages intersecting readable rows, including frozen points.
    pub active_page_bytes: u64,
    /// Requested budget; selection rounds down to whole base pages.
    pub budget_bytes: u64,
    pub target_dram_bytes: u64,
    pub primary_dram_bytes: u64,
    pub rerank_dram_bytes: u64,
    pub promoted_bytes: u64,
    pub demoted_bytes: u64,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Default, Clone)]
struct Bitmap(Vec<u64>);

impl Bitmap {
    fn insert(&mut self, id: usize) {
        if self.0.len() <= id / 64 {
            self.0.resize(id / 64 + 1, 0);
        }
        self.0[id / 64] |= 1 << (id % 64);
    }

    fn contains(&self, id: usize) -> bool {
        self.0
            .get(id / 64)
            .is_some_and(|word| word & (1 << (id % 64)) != 0)
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &word)| {
            let mut remaining = word;
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                Some(i * 64 + bit)
            })
        })
    }

    fn count(&self) -> u64 {
        self.0.iter().map(|word| u64::from(word.count_ones())).sum()
    }
}

fn priority(id: usize, seed: u64) -> u64 {
    let mut value = (id as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[derive(Debug)]
struct Selection {
    budget_pages: u64,
    seed: u64,
    active: Bitmap,
    heap: BinaryHeap<(u64, usize)>,
}

impl Selection {
    fn new(budget_pages: u64, seed: u64) -> Self {
        Self {
            budget_pages,
            seed,
            active: Bitmap::default(),
            heap: BinaryHeap::new(),
        }
    }

    fn selected(&self) -> Bitmap {
        let mut result = Bitmap::default();
        for &(_, id) in &self.heap {
            result.insert(id);
        }
        result
    }

    fn update(&mut self, active: Bitmap) -> (Vec<usize>, Vec<usize>) {
        let before = self.selected();
        // Insert-only batches process new pages incrementally. Deletion can expose
        // previously unselected candidates, so rebuild from the remaining live set.
        let removed = self.active.iter().any(|id| !active.contains(id));
        if removed {
            self.heap.clear();
        }
        for id in active
            .iter()
            .filter(|&id| removed || !self.active.contains(id))
        {
            let item = (priority(id, self.seed), id);
            if (self.heap.len() as u64) < self.budget_pages {
                self.heap.push(item);
            } else if let Some(mut worst) = self.heap.peek_mut() {
                if item < *worst {
                    *worst = item;
                }
            }
        }
        let after = self.selected();
        let demote = before.iter().filter(|&id| !after.contains(id)).collect();
        let promote = after.iter().filter(|&id| !before.contains(id)).collect();
        self.active = active;
        (demote, promote)
    }
}

// Page IDs interleave stable arena kinds: primary=0, rerank=1. IDs do not depend
// on reserved capacity, mapping addresses, or which batches published the rows.
fn live_pages(
    arenas: &[&Buffer],
    readable: impl Iterator<Item = usize>,
    page: usize,
) -> io::Result<(Bitmap, u64)> {
    if arenas.is_empty() || arenas.len() > 2 {
        return Err(io::Error::other("expected one or two vector arenas"));
    }
    let mut active = Bitmap::default();
    let mut bytes = 0_u64;
    for id in readable {
        for (kind, arena) in arenas.iter().enumerate() {
            if id >= arena.len() {
                return Err(io::Error::other("live slot outside vector arena"));
            }
            let stride = arena.stride().value();
            if stride == 0 {
                continue;
            }
            bytes = bytes
                .checked_add(stride as u64)
                .ok_or_else(|| io::Error::other("active byte count overflow"))?;
            let first = id * stride / page;
            let last = ((id + 1) * stride - 1) / page;
            for offset in first..=last {
                active.insert(offset * 2 + kind);
            }
        }
    }
    Ok((active, bytes))
}

#[derive(Debug)]
pub(crate) struct Controller {
    options: VectorPlacementOptions,
    page: usize,
    selection: Selection,
    last: PlacementReport,
    failed: bool,
}

impl Controller {
    pub(crate) fn new(options: VectorPlacementOptions) -> io::Result<Self> {
        let page = validate_platform(options)?;
        Ok(Self {
            options,
            page,
            selection: Selection::new(options.budget_bytes / page as u64, options.seed),
            last: PlacementReport::default(),
            failed: false,
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn assert_residency(&self, arenas: &[&Buffer]) {
        self.report().unwrap();
        let selected = self.selection.selected();
        let mut dram_pages = 0;
        for (kind, arena) in arenas.iter().enumerate() {
            let count = arena.allocated_bytes() / self.page;
            for first in (0..count).step_by(4096) {
                let end = count.min(first + 4096);
                let addresses: Vec<*const libc::c_void> = (first..end)
                    .map(|offset| {
                        // SAFETY: Every page address belongs to this live mapping.
                        unsafe { arena.as_ptr().add(offset * self.page).cast() }
                    })
                    .collect();
                let mut status = vec![-libc::EIO; addresses.len()];
                // SAFETY: Query only; all arrays cover the requested page count.
                let result = unsafe {
                    libc::syscall(
                        libc::SYS_move_pages,
                        0_i32,
                        addresses.len(),
                        addresses.as_ptr(),
                        std::ptr::null::<i32>(),
                        status.as_mut_ptr(),
                        0_u32,
                    )
                };
                assert_eq!(result, 0, "{}", io::Error::last_os_error());
                for (offset, node) in (first..end).zip(status) {
                    let chosen = selected.contains(offset * 2 + kind);
                    if chosen {
                        assert_eq!(node, self.options.dram_node as i32);
                        dram_pages += 1;
                    } else {
                        assert!(
                            node == self.options.cxl_node as i32 || node == -libc::ENOENT,
                            "unexpected page node {node}"
                        );
                    }
                }
            }
        }
        assert_eq!(dram_pages * self.page as u64, self.last.target_dram_bytes);
    }

    pub(crate) fn report(&self) -> io::Result<PlacementReport> {
        if self.failed {
            return Err(io::Error::other(
                "vector placement previously failed; reload the index",
            ));
        }
        Ok(self.last)
    }

    pub(crate) fn rebalance(
        &mut self,
        arenas: &[&Buffer],
        readable: impl Iterator<Item = usize>,
    ) -> io::Result<PlacementReport> {
        self.report()?;
        let start = Instant::now();
        let (active, bytes) = live_pages(arenas, readable, self.page)?;
        let page = self.page;
        self.apply(active, bytes, start, |ids, node| {
            move_pages(arenas, page, ids, node)
        })
    }

    fn apply(
        &mut self,
        active: Bitmap,
        bytes: u64,
        start: Instant,
        mut migrate: impl FnMut(&[usize], u32) -> io::Result<()>,
    ) -> io::Result<PlacementReport> {
        self.report()?;
        let active_page_bytes = active.count() * self.page as u64;
        let (demote, promote) = self.selection.update(active);
        // No budget headroom is needed, including when deleting selected pages.
        if let Err(error) = migrate(&demote, self.options.cxl_node)
            .and_then(|()| migrate(&promote, self.options.dram_node))
        {
            self.failed = true; // A syscall may already have moved some pages.
            return Err(error);
        }
        let primary = self
            .selection
            .heap
            .iter()
            .filter(|(_, id)| id % 2 == 0)
            .count() as u64;
        let total = self.selection.heap.len() as u64;
        let page = self.page as u64;
        self.last = PlacementReport {
            page_bytes: page,
            active_bytes: bytes,
            active_page_bytes,
            budget_bytes: self.options.budget_bytes,
            target_dram_bytes: total * page,
            primary_dram_bytes: primary * page,
            rerank_dram_bytes: (total - primary) * page,
            promoted_bytes: promote.len() as u64 * page,
            demoted_bytes: demote.len() as u64 * page,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        };
        Ok(self.last)
    }
}

#[cfg(target_os = "linux")]
fn validate_platform(options: VectorPlacementOptions) -> io::Result<usize> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    if options.dram_node == options.cxl_node
        || options.dram_node >= libc::c_ulong::BITS
        || options.cxl_node >= libc::c_ulong::BITS
    {
        return Err(invalid(
            "vector placement requires distinct NUMA nodes in 0..64",
        ));
    }
    // Size the query for the kernel's possible-node mask, even when both requested
    // nodes fit the single-word mbind mask used by Buffer.
    let possible = std::fs::read_to_string("/sys/devices/system/node/possible")?;
    let highest: usize = possible
        .trim()
        .rsplit([',', '-'])
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| invalid("invalid possible NUMA node list"))?;
    let bits = libc::c_ulong::BITS as usize;
    let mut allowed = vec![0 as libc::c_ulong; (highest + 1).div_ceil(bits)];
    // SAFETY: allowed covers maxnode bits; MPOL_F_MEMS_ALLOWED needs no address.
    if unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            std::ptr::null_mut::<libc::c_int>(),
            allowed.as_mut_ptr(),
            allowed.len() * bits,
            std::ptr::null::<u8>(),
            4_u32,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    for node in [options.dram_node as usize, options.cxl_node as usize] {
        if node > highest || allowed[node / bits] & (1 << (node % bits)) == 0 {
            return Err(invalid(
                "vector placement node is outside the allowed memory nodes",
            ));
        }
    }
    // SAFETY: sysconf has no memory preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 || !(page as usize).is_power_of_two() {
        return Err(io::Error::other("invalid base page size"));
    }
    Ok(page as usize)
}

#[cfg(not(target_os = "linux"))]
fn validate_platform(_: VectorPlacementOptions) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vector placement requires Linux",
    ))
}

#[cfg(target_os = "linux")]
fn move_pages(arenas: &[&Buffer], page: usize, ids: &[usize], node: u32) -> io::Result<()> {
    for chunk in ids.chunks(4096) {
        let mut addresses: Vec<*const libc::c_void> = chunk
            .iter()
            .map(|&id| {
                // SAFETY: IDs were generated from rows in these live, owned mappings.
                unsafe { arenas[id % 2].as_ptr().add((id / 2) * page).cast() }
            })
            .collect();
        move_batch(&mut addresses, node, |addresses, status| {
            let nodes = vec![node as libc::c_int; addresses.len()];
            // SAFETY: buffers cover count entries; mappings remain alive for this call.
            let result = unsafe {
                libc::syscall(
                    libc::SYS_move_pages,
                    0_i32,
                    addresses.len(),
                    addresses.as_ptr(),
                    nodes.as_ptr(),
                    status.as_mut_ptr(),
                    2_u32,
                )
            }; // MPOL_MF_MOVE
            if result < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        })?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn move_batch<T: Copy>(
    addresses: &mut Vec<T>,
    node: u32,
    mut syscall: impl FnMut(&[T], &mut [i32]) -> io::Result<()>,
) -> io::Result<()> {
    for attempt in 0..3 {
        if addresses.is_empty() {
            return Ok(());
        }
        let mut status = vec![-libc::EIO; addresses.len()];
        syscall(addresses, &mut status)?;
        let mut retry = 0;
        for (i, result) in status.into_iter().enumerate() {
            if result == node as i32 {
                continue;
            }
            if attempt < 2 && (result == -libc::EBUSY || result == -libc::EAGAIN) {
                addresses[retry] = addresses[i];
                retry += 1;
            } else {
                let error =
                    io::Error::from_raw_os_error(if result < 0 { -result } else { libc::EIO });
                return Err(io::Error::new(
                    error.kind(),
                    format!("vector page migration to node {node}: {error}"),
                ));
            }
        }
        addresses.truncate(retry);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn move_pages(_: &[&Buffer], _: usize, ids: &[usize], _: u32) -> io::Result<()> {
    if ids.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "vector placement requires Linux",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{Align, Bytes};

    fn bitmap(ids: impl IntoIterator<Item = usize>) -> Bitmap {
        let mut result = Bitmap::default();
        for id in ids {
            result.insert(id);
        }
        result
    }

    #[test]
    fn selection_is_global_reproducible_and_batch_independent() {
        let mut one = Selection::new(19, 52);
        let mut batches = Selection::new(19, 52);
        one.update(bitmap(0..400));
        for end in [1, 7, 59, 127, 400] {
            batches.update(bitmap(0..end));
        }
        assert_eq!(one.selected().0, batches.selected().0);
        assert_eq!(one.heap.len(), 19);
        let mut expected: Vec<_> = (0..400).map(|id| (priority(id, 52), id)).collect();
        expected.sort_unstable();
        assert_eq!(
            one.selected().0,
            bitmap(expected[..19].iter().map(|x| x.1)).0
        );
        let mut different = Selection::new(19, 53);
        different.update(bitmap(0..400));
        assert_ne!(one.selected().0, different.selected().0);
    }

    #[test]
    fn deletion_refills_and_slot_reuse_restores_sample() {
        let mut selection = Selection::new(10, 0);
        selection.update(bitmap(0..100));
        let original = selection.selected();
        let retired = original.iter().next().unwrap();
        let (demote, promote) = selection.update(bitmap((0..100).filter(|&id| id != retired)));
        assert_eq!(demote, vec![retired]);
        assert_eq!(promote.len(), 1);
        selection.update(bitmap(0..100));
        assert_eq!(selection.selected().0, original.0);
        let (demote, promote) = selection.update(Bitmap::default());
        assert_eq!(demote.len(), 10);
        assert!(promote.is_empty());
    }

    #[test]
    fn sparse_rows_shared_pages_and_frozen_tail() {
        let primary = Buffer::new(101, Bytes::new(64), Align::_128).unwrap();
        let rerank = Buffer::new(101, Bytes::new(8192), Align::_128).unwrap();
        let (active, bytes) =
            live_pages(&[&primary, &rerank], [0, 1, 100].into_iter(), 4096).unwrap();
        assert_eq!(bytes, 3 * (64 + 8192));
        assert_eq!(active.count(), 8); // Two primary pages; six reranking pages.
        assert_eq!(
            active.iter().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 5, 7, 401, 403]
        );
        assert!(live_pages(&[&primary], [101].into_iter(), 4096).is_err());
    }

    fn controller(budget: u64) -> Controller {
        Controller {
            options: VectorPlacementOptions {
                budget_bytes: budget,
                ..Default::default()
            },
            page: 4096,
            selection: Selection::new(budget / 4096, 0),
            last: PlacementReport::default(),
            failed: false,
        }
    }

    #[test]
    fn budgets_round_down_and_report_shared_arenas() {
        for (budget, count) in [(0, 0), (4095, 0), (4097, 1), (8 * 4096, 8), (u64::MAX, 10)] {
            let mut controller = controller(budget);
            let report = controller
                .apply(bitmap(0..10), 123, Instant::now(), |_, _| Ok(()))
                .unwrap();
            assert_eq!(report.budget_bytes, budget);
            assert_eq!(report.target_dram_bytes, count * 4096);
            assert_eq!(
                report.target_dram_bytes,
                report.primary_dram_bytes + report.rerank_dram_bytes
            );
            assert_eq!(report.active_page_bytes, 10 * 4096);
        }
    }

    #[test]
    fn demotion_precedes_promotion_and_failure_poisoning() {
        let mut controller = controller(4096);
        controller
            .apply(bitmap([0]), 4096, Instant::now(), |_, _| Ok(()))
            .unwrap();
        let mut calls = Vec::new();
        let failure = controller.apply(bitmap([1]), 4096, Instant::now(), |ids, node| {
            calls.push((ids.to_vec(), node));
            if node == 0 {
                Err(io::Error::other("injected partial migration"))
            } else {
                Ok(())
            }
        });
        assert!(failure.is_err());
        assert_eq!(calls, vec![(vec![0], 1), (vec![1], 0)]);
        assert!(controller.report().is_err());
        assert!(
            controller
                .apply(bitmap([1]), 4096, Instant::now(), |_, _| panic!(
                    "must not migrate after failure"
                ))
                .is_err()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn per_page_migration_results_and_bounded_retries() {
        let mut calls = 0;
        let mut addresses = vec![0, 1, 2];
        move_batch(&mut addresses, 0, |ids, status| {
            calls += 1;
            if calls == 1 {
                status.copy_from_slice(&[0, -libc::EBUSY, -libc::EAGAIN]);
            } else {
                assert_eq!(ids, &[1, 2]);
                status.fill(0);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 2);
        for failure in [-libc::ENOMEM, -libc::ENOENT, 1] {
            assert!(
                move_batch(&mut vec![0], 0, |_, status| {
                    status.fill(failure);
                    Ok(())
                })
                .is_err()
            );
        }
        calls = 0;
        assert!(
            move_batch(&mut vec![0], 0, |_, status| {
                calls += 1;
                status.fill(-libc::EBUSY);
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 3);
        assert!(
            move_batch(&mut vec![0], 0, |_, _| Err(io::Error::other(
                "syscall failed"
            )))
            .is_err()
        );
    }

    #[test]
    fn invalid_nodes_fail_before_allocation() {
        for (dram_node, cxl_node) in [(0, 0), (u32::MAX, 1), (0, u32::MAX)] {
            assert!(
                Controller::new(VectorPlacementOptions {
                    dram_node,
                    cxl_node,
                    ..Default::default()
                })
                .is_err()
            );
        }
    }
}
