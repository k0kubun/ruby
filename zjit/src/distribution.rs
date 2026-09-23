//! Type frequency distribution tracker.

use crate::options::NumProfiles;

/// This implementation was inspired by the type feedback module from Google's S6, which was
/// written in C++ for use with Python. This is a new implementation in Rust created for use with
/// Ruby instead of Python.
#[derive(Debug, Clone)]
pub struct Distribution<T: Copy + PartialEq + Default, const N: usize> {
    /// buckets and counts have the same length
    /// `buckets[0]` is always the most common item
    buckets: [T; N],
    counts: [NumProfiles; N],
    /// if there is no more room, increment the fallback
    other: NumProfiles,
    // TODO(max): Add count disparity, which can help determine when to reset the distribution
}

impl<T: Copy + PartialEq + Default, const N: usize> Distribution<T, N> {
    pub fn new() -> Self {
        Self { buckets: [Default::default(); N], counts: [0; N], other: 0 }
    }

    pub fn observe(&mut self, item: T) {
        for (bucket, count) in self.buckets.iter_mut().zip(self.counts.iter_mut()) {
            if *bucket == item || *count == 0 {
                *bucket = item;
                *count = count.saturating_add(1);
                // Keep the most frequent item at the front
                self.bubble_up();
                return;
            }
        }
        self.other = self.other.saturating_add(1);
    }

    /// Count an item without recording it, as if all buckets were taken by other items.
    pub fn observe_other(&mut self) {
        self.other = self.other.saturating_add(1);
    }

    /// Keep the highest counted bucket at index 0
    fn bubble_up(&mut self) {
        if N == 0 { return; }
        let max_index = self.counts.into_iter().enumerate().max_by_key(|(_, val)| *val).unwrap().0;
        if max_index != 0 {
            self.counts.swap(0, max_index);
            self.buckets.swap(0, max_index);
        }
    }

    pub fn each_item(&self) -> impl Iterator<Item = T> + '_ {
        self.buckets.iter().zip(self.counts.iter())
            .filter_map(|(&bucket, &count)| if count > 0 { Some(bucket) } else { None })
    }

    pub fn each_item_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.buckets.iter_mut().zip(self.counts.iter())
            .filter_map(|(bucket, &count)| if count > 0 { Some(bucket) } else { None })
    }

    /// Remove every item for which `drop_p` returns true. Their counts are folded into
    /// `other`, so the distribution still reflects that more kinds of items were seen.
    /// The remaining items are compacted toward index 0 because [Self::observe] stops
    /// at the first empty bucket.
    pub fn drop_items(&mut self, mut drop_p: impl FnMut(T) -> bool) {
        let mut kept = 0;
        for i in 0..N {
            let count = self.counts[i];
            if count == 0 {
                continue;
            }
            if drop_p(self.buckets[i]) {
                self.other = self.other.saturating_add(count);
            } else {
                self.buckets[kept] = self.buckets[i];
                self.counts[kept] = count;
                kept += 1;
            }
        }
        for i in kept..N {
            self.buckets[i] = Default::default();
            self.counts[i] = 0;
        }
        // Keep the most frequent item at the front. If the front item was dropped, move the
        // first most frequent item there, preserving the order of the others.
        if let Some(max_index) = (0..kept).reduce(|max, i| if self.counts[i] > self.counts[max] { i } else { max }) {
            self.buckets[..=max_index].rotate_right(1);
            self.counts[..=max_index].rotate_right(1);
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum DistributionKind {
    /// No types seen
    Empty,
    /// One type seen
    Monomorphic,
    /// Between 2 and (fixed) N types seen
    Polymorphic,
    /// Polymorphic, but with a significant skew towards one type
    SkewedPolymorphic,
    /// More than N types seen with no clear winner
    Megamorphic,
    /// Megamorphic, but with a significant skew towards one type
    SkewedMegamorphic,
}

#[derive(Debug, Clone)]
pub struct DistributionSummary<T: Copy + PartialEq + Default + std::fmt::Debug, const N: usize> {
    kind: DistributionKind,
    buckets: [T; N],
    // TODO(max): Determine if we need some notion of stability
}

const SKEW_THRESHOLD: f64 = 0.75;

impl<T: Copy + PartialEq + Default + std::fmt::Debug, const N: usize> DistributionSummary<T, N> {
    pub fn empty() -> Self {
        Self { kind: DistributionKind::Empty, buckets: [Default::default(); N] }
    }

    pub fn new(dist: &Distribution<T, N>) -> Self {
        #[cfg(debug_assertions)]
        {
            let first_count = dist.counts[0];
            for &count in &dist.counts[1..] {
                assert!(first_count >= count, "First count should be the largest");
            }
        }
        let num_seen = dist.counts.iter().map(|&c| usize::from(c)).sum::<usize>() + usize::from(dist.other);
        let kind = if dist.other == 0 {
            // Seen <= N types total
            if dist.counts[0] == 0 {
                DistributionKind::Empty
            } else if dist.counts[1] == 0 {
                DistributionKind::Monomorphic
            } else if (dist.counts[0] as f64)/(num_seen as f64) >= SKEW_THRESHOLD {
                DistributionKind::SkewedPolymorphic
            } else {
                DistributionKind::Polymorphic
            }
        } else {
            // Seen > N types total; considered megamorphic
            if (dist.counts[0] as f64)/(num_seen as f64) >= SKEW_THRESHOLD {
                DistributionKind::SkewedMegamorphic
            } else {
                DistributionKind::Megamorphic
            }
        };
        Self { kind, buckets: dist.buckets }
    }

    pub fn is_monomorphic(&self) -> bool {
        self.kind == DistributionKind::Monomorphic
    }

    pub fn is_polymorphic(&self) -> bool {
        self.kind == DistributionKind::Polymorphic
    }

    pub fn is_skewed_polymorphic(&self) -> bool {
        self.kind == DistributionKind::SkewedPolymorphic
    }

    pub fn is_megamorphic(&self) -> bool {
        self.kind == DistributionKind::Megamorphic
    }

    pub fn is_skewed_megamorphic(&self) -> bool {
        self.kind == DistributionKind::SkewedMegamorphic
    }

    pub fn bucket(&self, idx: usize) -> T {
        assert!(idx < N, "index {idx} out of bounds for buckets[{N}]");
        self.buckets[idx]
    }

    pub fn buckets(&self) -> &[T] {
        &self.buckets
    }
}

#[cfg(test)]
mod distribution_tests {
    use super::*;

    #[test]
    fn start_empty() {
        let dist = Distribution::<usize, 4>::new();
        assert_eq!(dist.other, 0);
        assert!(dist.counts.iter().all(|&b| b == 0));
    }

    #[test]
    fn observe_adds_record() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        assert_eq!(dist.buckets[0], 10);
        assert_eq!(dist.counts[0], 1);
        assert_eq!(dist.other, 0);
    }

    #[test]
    fn observe_increments_record() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        assert_eq!(dist.buckets[0], 10);
        assert_eq!(dist.counts[0], 2);
        assert_eq!(dist.other, 0);
    }

    #[test]
    fn observe_two() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        dist.observe(11);
        assert_eq!(dist.buckets[0], 11);
        assert_eq!(dist.counts[0], 3);
        assert_eq!(dist.buckets[1], 10);
        assert_eq!(dist.counts[1], 2);
        assert_eq!(dist.other, 0);
    }

    #[test]
    fn observe_with_max_increments_other() {
        let mut dist = Distribution::<usize, 0>::new();
        dist.observe(10);
        assert!(dist.buckets.is_empty());
        assert!(dist.counts.is_empty());
        assert_eq!(dist.other, 1);
    }

    fn dist_from(buckets: [usize; 4], counts: [NumProfiles; 4], other: NumProfiles) -> Distribution<usize, 4> {
        Distribution { buckets, counts, other }
    }

    #[test]
    fn drop_items_compacts_and_folds_into_other() {
        let mut dist = dist_from([12, 11, 10, 13], [3, 2, 1, 1], 0);
        dist.drop_items(|item| item == 11);
        assert_eq!(dist.buckets, [12, 10, 13, 0]);
        assert_eq!(dist.counts, [3, 1, 1, 0]);
        assert_eq!(dist.other, 2);
    }

    #[test]
    fn drop_items_without_match_is_noop() {
        let mut dist = dist_from([12, 11, 10, 13], [3, 3, 1, 3], 1);
        dist.drop_items(|_| false);
        assert_eq!(dist.buckets, [12, 11, 10, 13]);
        assert_eq!(dist.counts, [3, 3, 1, 3]);
        assert_eq!(dist.other, 1);
    }

    #[test]
    fn drop_items_moves_most_frequent_item_to_front() {
        let mut dist = dist_from([10, 11, 12, 13], [3, 1, 2, 2], 0);
        dist.drop_items(|item| item == 10);
        // 12 and 13 tie; the first one moves to the front and the others keep their order
        assert_eq!(dist.buckets, [12, 11, 13, 0]);
        assert_eq!(dist.counts, [2, 1, 2, 0]);
        assert_eq!(dist.other, 3);
        DistributionSummary::new(&dist); // asserts counts[0] is the largest in debug builds
    }

    #[test]
    fn drop_items_does_not_duplicate_buckets() {
        let mut dist = dist_from([10, 11, 12, 0], [3, 1, 1, 0], 0);
        dist.drop_items(|item| item == 11);
        // A hole at index 1 would make observe() store a second bucket for 12
        dist.observe(12);
        assert_eq!(dist.buckets, [10, 12, 0, 0]);
        assert_eq!(dist.counts, [3, 2, 0, 0]);
        assert_eq!(dist.other, 1);
        dist.observe(14);
        assert_eq!(dist.buckets, [10, 12, 14, 0]);
        assert_eq!(dist.counts, [3, 2, 1, 0]);
    }

    #[test]
    fn drop_items_keeps_site_non_monomorphic() {
        let mut dist = dist_from([10, 11, 12, 13], [1, 1, 1, 1], 1);
        dist.drop_items(|item| item != 10);
        assert_eq!(dist.buckets, [10, 0, 0, 0]);
        assert_eq!(dist.counts, [1, 0, 0, 0]);
        assert_eq!(dist.other, 4);
        assert_eq!(DistributionSummary::new(&dist).kind, DistributionKind::Megamorphic);

        // Dropping everything leaves no buckets but keeps the evidence
        dist.drop_items(|_| true);
        assert_eq!(dist.each_item().count(), 0);
        assert_eq!(dist.counts, [0, 0, 0, 0]);
        assert_eq!(dist.other, 5);
        assert_eq!(DistributionSummary::new(&dist).kind, DistributionKind::Megamorphic);
    }

    #[test]
    fn drop_items_after_monomorphic_is_not_monomorphic() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(11);
        dist.drop_items(|item| item == 11);
        assert_eq!(dist.buckets, [10, 0, 0, 0]);
        assert_eq!(dist.other, 1);
        assert!(!DistributionSummary::new(&dist).is_monomorphic());
    }

    #[test]
    fn empty_distribution_returns_empty_summary() {
        let dist = Distribution::<usize, 4>::new();
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::Empty);
    }

    #[test]
    fn monomorphic_distribution_returns_monomorphic_summary() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::Monomorphic);
        assert_eq!(summary.buckets[0], 10);
    }

    #[test]
    fn polymorphic_distribution_returns_polymorphic_summary() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::Polymorphic);
        assert_eq!(summary.buckets[0], 11);
        assert_eq!(summary.buckets[1], 10);
    }

    #[test]
    fn skewed_polymorphic_distribution_returns_skewed_polymorphic_summary() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        dist.observe(11);
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::SkewedPolymorphic);
        assert_eq!(summary.buckets[0], 11);
        assert_eq!(summary.buckets[1], 10);
    }

    #[test]
    fn megamorphic_distribution_returns_megamorphic_summary() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(11);
        dist.observe(12);
        dist.observe(13);
        dist.observe(14);
        dist.observe(11);
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::Megamorphic);
        assert_eq!(summary.buckets[0], 11);
    }

    #[test]
    fn skewed_megamorphic_distribution_returns_skewed_megamorphic_summary() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(12);
        dist.observe(13);
        dist.observe(14);
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.kind, DistributionKind::SkewedMegamorphic);
        assert_eq!(summary.buckets[0], 12);
    }
}
