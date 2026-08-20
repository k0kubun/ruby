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

    /// Keep the highest counted bucket at index 0
    fn bubble_up(&mut self) {
        if N == 0 { return; }
        let max_index = self.counts.into_iter().enumerate().max_by_key(|(_, val)| *val).unwrap().0;
        if max_index != 0 {
            self.counts.swap(0, max_index);
            self.buckets.swap(0, max_index);
        }
    }

    /// Look `item` up and, if `matches` finds no bucket holding it and a bucket is free, record
    /// it there. Unlike [`Self::observe`] this does not count the sighting and does not reorder
    /// the buckets, so the index it reports stays valid for as long as the distribution is only
    /// updated this way: a caller that remembers how many buckets were occupied at some earlier
    /// point can tell from the index alone whether an item was already known then.
    ///
    /// New buckets always start at a count of 1, which cannot exceed `counts[0]`, so this
    /// preserves "`buckets[0]` is the most common item" without bubbling.
    pub fn observe_stable(&mut self, item: T, matches: impl Fn(T, T) -> bool) -> StableBucket {
        for (index, (bucket, count)) in self.buckets.iter_mut().zip(self.counts.iter_mut()).enumerate() {
            if *count == 0 {
                *bucket = item;
                *count = 1;
                return StableBucket::Inserted(index);
            }
            if matches(*bucket, item) {
                return StableBucket::Existing(index);
            }
        }
        self.other = self.other.saturating_add(1);
        StableBucket::Full
    }

    /// Drop every bucket but bucket 0, and forget the items that did not fit in one, leaving the
    /// distribution as if only bucket 0's item had ever been observed.
    ///
    /// [`crate::profile::IseqProfile::observe_ivar_fallback`] is the only caller: a profile whose
    /// buckets are all spoken for by shapes an ISEQ saw at boot cannot record the shape that would
    /// fix a site missing now, so the choice is between forgetting the cold ones and never
    /// specializing the site again.
    pub fn retain_primary(&mut self) {
        for index in 1..N {
            self.buckets[index] = Default::default();
            self.counts[index] = 0;
        }
        self.other = 0;
    }

    /// The item in bucket `idx`. Zero-count buckets hold `T::default()`.
    pub fn bucket(&self, idx: usize) -> T {
        assert!(idx < N, "index {idx} out of bounds for buckets[{N}]");
        self.buckets[idx]
    }

    /// How many times bucket `idx`'s item was observed.
    pub fn count(&self, idx: usize) -> NumProfiles {
        if idx < N { self.counts[idx] } else { 0 }
    }

    /// Total number of observations, bucketed or not.
    pub fn num_observed(&self) -> usize {
        usize::from(self.other) + self.counts.iter().map(|&count| usize::from(count)).sum::<usize>()
    }

    /// Every item in a non-empty bucket, bucket 0 first.
    /// How many buckets hold an observed item. Zero means nothing was observed;
    /// `N` plus a non-zero `other` means megamorphic.
    pub fn num_buckets_used(&self) -> usize {
        self.counts.iter().filter(|&&count| count > 0).count()
    }

    pub fn each_item(&self) -> impl Iterator<Item = T> + '_ {
        self.buckets.iter().zip(self.counts.iter())
            .filter_map(|(&bucket, &count)| if count > 0 { Some(bucket) } else { None })
    }

    pub fn each_item_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.buckets.iter_mut().zip(self.counts.iter())
            .filter_map(|(bucket, &count)| if count > 0 { Some(bucket) } else { None })
    }
}

/// Where an item sits in a [`Distribution`] after [`Distribution::observe_stable`].
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum StableBucket {
    /// The item already had this bucket.
    Existing(usize),
    /// The item was recorded in this bucket, which was empty until now.
    Inserted(usize),
    /// Every bucket belongs to some other item; this one was only counted as `other`.
    Full,
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
    /// How many samples landed in each bucket, in the same order as `buckets`
    counts: [NumProfiles; N],
    /// How many samples did not fit in any bucket
    other: NumProfiles,
    // TODO(max): Determine if we need some notion of stability
}

const SKEW_THRESHOLD: f64 = 0.75;

impl<T: Copy + PartialEq + Default + std::fmt::Debug, const N: usize> DistributionSummary<T, N> {
    pub fn empty() -> Self {
        Self { kind: DistributionKind::Empty, buckets: [Default::default(); N], counts: [0; N], other: 0 }
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
        Self { kind, buckets: dist.buckets, counts: dist.counts, other: dist.other }
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

    /// How many samples landed in `buckets[idx]`. 0 means the bucket is unused.
    pub fn bucket_count(&self, idx: usize) -> NumProfiles {
        assert!(idx < N, "index {idx} out of bounds for buckets[{N}]");
        self.counts[idx]
    }

    /// Total number of samples this summary was built from, including the ones that did not fit
    /// in a bucket.
    pub fn num_seen(&self) -> u32 {
        self.counts.iter().map(|&c| u32::from(c)).sum::<u32>() + u32::from(self.other)
    }

    /// Fraction of observed samples that landed in the buckets for which `keep` returns true.
    /// Used to decide whether a guard chain over those buckets is worth building: samples in
    /// other buckets, and samples that did not fit in any bucket, have to take the fallback.
    pub fn coverage(&self, keep: impl Fn(usize, T) -> bool) -> f64 {
        let num_seen = self.num_seen();
        if num_seen == 0 { return 0.0; }
        let covered: u32 = self.counts.iter().enumerate()
            .filter(|&(idx, &count)| count > 0 && keep(idx, self.buckets[idx]))
            .map(|(_, &count)| u32::from(count))
            .sum();
        (covered as f64) / (num_seen as f64)
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

    #[test]
    fn observe_stable_reports_new_and_known_buckets() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        // A bucket that already exists is reported without being counted again.
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Existing(0));
        assert_eq!(dist.counts[0], 1);
        // New items land past the buckets that were occupied, which is how a caller tells
        // them apart from the ones the compiled code already knows about.
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Inserted(1));
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Existing(1));
        assert_eq!(dist.observe_stable(12, |a, b| a == b), StableBucket::Inserted(2));
    }

    #[test]
    fn observe_stable_does_not_reorder_buckets() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        for _ in 0..10 {
            dist.observe_stable(11, |a, b| a == b);
        }
        // `observe` would have bubbled 11 to the front; a stable observation may not, or the
        // index it reported earlier would now name a different item.
        assert_eq!(dist.buckets[0], 10);
        assert_eq!(dist.buckets[1], 11);
        // Counts stay ordered, so DistributionSummary's invariant still holds.
        assert!(dist.counts[0] >= dist.counts[1]);
    }

    #[test]
    fn observe_stable_reports_full_without_recording() {
        let mut dist = Distribution::<usize, 2>::new();
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Inserted(0));
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Inserted(1));
        assert_eq!(dist.observe_stable(12, |a, b| a == b), StableBucket::Full);
        assert_eq!(dist.other, 1);
        assert_eq!(dist.buckets, [10, 11]);
    }

    #[test]
    fn observe_stable_matches_with_the_given_predicate() {
        let mut dist = Distribution::<(usize, usize), 4>::new();
        dist.observe_stable((1, 100), |a, b| a.0 == b.0);
        // Equal under the predicate, different as a whole: still the same bucket.
        assert_eq!(dist.observe_stable((1, 999), |a, b| a.0 == b.0), StableBucket::Existing(0));
        assert_eq!(dist.buckets[0], (1, 100));
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

    #[test]
    fn summary_exposes_counts_and_num_seen() {
        let mut dist = Distribution::<usize, 2>::new();
        dist.observe(10);
        dist.observe(10);
        dist.observe(11);
        dist.observe(12); // does not fit; counted in other
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.bucket_count(0), 2);
        assert_eq!(summary.bucket_count(1), 1);
        assert_eq!(summary.num_seen(), 4);
    }

    #[test]
    fn coverage_counts_only_kept_buckets() {
        let mut dist = Distribution::<usize, 2>::new();
        dist.observe(10);
        dist.observe(10);
        dist.observe(11);
        dist.observe(12); // does not fit; counted in other
        let summary = DistributionSummary::new(&dist);
        assert_eq!(summary.coverage(|_, _| true), 0.75);
        assert_eq!(summary.coverage(|idx, _| idx == 0), 0.5);
        assert_eq!(summary.coverage(|_, item| item == 11), 0.25);
        assert_eq!(summary.coverage(|_, _| false), 0.0);
    }

    #[test]
    fn empty_summary_has_no_coverage() {
        let summary = DistributionSummary::<usize, 4>::empty();
        assert_eq!(summary.num_seen(), 0);
        assert_eq!(summary.coverage(|_, _| true), 0.0);
    }
}
