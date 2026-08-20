//! Type frequency distribution tracker.

use crate::options::NumProfiles;

/// Buckets 1 through `N-1` of a [`Distribution`], allocated only once a second
/// distinct item shows up. Index 0 is unused so that bucket numbering lines up
/// with `Distribution`'s.
#[derive(Debug, Clone)]
struct Tail<T: Copy + PartialEq + Default, const N: usize> {
    buckets: [T; N],
    counts: [NumProfiles; N],
}

impl<T: Copy + PartialEq + Default, const N: usize> Tail<T, N> {
    fn new() -> Self {
        Self { buckets: [Default::default(); N], counts: [0; N] }
    }
}

/// This implementation was inspired by the type feedback module from Google's S6, which was
/// written in C++ for use with Python. This is a new implementation in Rust created for use with
/// Ruby instead of Python.
///
/// Bucket 0 is stored inline and buckets 1..N are boxed, because ZJIT retains
/// one of these per profiled operand of every profiled instruction and the
/// overwhelming majority of them only ever see a single type. On an
/// RDoc-over-stdlib workload, 98% of 648K live distributions were monomorphic,
/// so an inline `[T; N]` spent 30 MB storing zeroes.
#[derive(Debug, Clone)]
pub struct Distribution<T: Copy + PartialEq + Default, const N: usize> {
    /// Most frequently observed item, i.e. bucket 0. Meaningful only when
    /// `primary_count` is non-zero.
    primary: T,
    primary_count: NumProfiles,
    /// if there is no more room, increment the fallback
    other: NumProfiles,
    /// Buckets 1..N, or `None` while this distribution is monomorphic.
    tail: Option<Box<Tail<T, N>>>,
    // TODO(max): Add count disparity, which can help determine when to reset the distribution
}

impl<T: Copy + PartialEq + Default, const N: usize> Distribution<T, N> {
    pub fn new() -> Self {
        Self { primary: Default::default(), primary_count: 0, other: 0, tail: None }
    }

    /// The item in bucket `idx`. Zero-count buckets hold `T::default()`.
    pub fn bucket(&self, idx: usize) -> T {
        assert!(idx < N, "index {idx} out of bounds for buckets[{N}]");
        if idx == 0 {
            self.primary
        } else {
            self.tail.as_ref().map_or_else(Default::default, |tail| tail.buckets[idx])
        }
    }

    /// How many times bucket `idx`'s item was observed.
    pub fn count(&self, idx: usize) -> NumProfiles {
        if idx == 0 {
            self.primary_count
        } else if idx < N {
            self.tail.as_ref().map_or(0, |tail| tail.counts[idx])
        } else {
            0
        }
    }

    /// Observations that did not fit in any bucket.
    pub fn other(&self) -> NumProfiles {
        self.other
    }

    /// Total number of observations, bucketed or not.
    pub fn num_observed(&self) -> usize {
        let mut total = usize::from(self.primary_count) + usize::from(self.other);
        if let Some(tail) = self.tail.as_ref() {
            total += tail.counts[1..].iter().map(|&count| usize::from(count)).sum::<usize>();
        }
        total
    }

    /// How many buckets hold an observed item. Zero means nothing was observed;
    /// `N` alongside a non-zero [`Self::other`] means megamorphic.
    pub fn num_buckets_used(&self) -> usize {
        (0..N).filter(|&idx| self.count(idx) > 0).count()
    }

    /// Bytes this distribution owns on the heap outside of itself, i.e. the
    /// boxed tail once it has gone polymorphic.
    pub fn heap_size(&self) -> usize {
        self.tail.as_ref().map_or(0, |_| size_of::<Tail<T, N>>())
    }

    /// Snapshot of every bucket, for [`DistributionSummary`].
    fn buckets(&self) -> [T; N] {
        let mut buckets = [Default::default(); N];
        for (idx, bucket) in buckets.iter_mut().enumerate() {
            *bucket = self.bucket(idx);
        }
        buckets
    }

    pub fn observe(&mut self, item: T) {
        if N == 0 {
            self.other = self.other.saturating_add(1);
            return;
        }
        // Bucket 0 takes the item if it already holds it or is still empty.
        if self.primary_count == 0 || self.primary == item {
            self.primary = item;
            self.primary_count = self.primary_count.saturating_add(1);
            self.bubble_up();
            return;
        }
        // A second distinct item: materialize buckets 1..N.
        let tail = self.tail.get_or_insert_with(|| Box::new(Tail::new()));
        for idx in 1..N {
            if tail.buckets[idx] == item || tail.counts[idx] == 0 {
                tail.buckets[idx] = item;
                tail.counts[idx] = tail.counts[idx].saturating_add(1);
                // Keep the most frequent item at the front
                self.bubble_up();
                return;
            }
        }
        self.other = self.other.saturating_add(1);
    }

    /// Keep the highest counted bucket at index 0. Ties go to the highest
    /// bucket index, matching what `Iterator::max_by_key` used to pick.
    fn bubble_up(&mut self) {
        if N == 0 { return; }
        let Some(tail) = self.tail.as_mut() else { return };
        let mut max_index = 0;
        let mut max_count = self.primary_count;
        for idx in 1..N {
            if tail.counts[idx] >= max_count {
                max_count = tail.counts[idx];
                max_index = idx;
            }
        }
        if max_index != 0 {
            std::mem::swap(&mut self.primary, &mut tail.buckets[max_index]);
            std::mem::swap(&mut self.primary_count, &mut tail.counts[max_index]);
        }
    }

    pub fn each_item(&self) -> impl Iterator<Item = T> + '_ {
        (0..N).filter_map(|idx| if self.count(idx) > 0 { Some(self.bucket(idx)) } else { None })
    }

    pub fn each_item_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        let primary = if N > 0 && self.primary_count > 0 { Some(&mut self.primary) } else { None };
        let tail = self.tail.as_mut().map(|tail| {
            let Tail { buckets, counts } = &mut **tail;
            buckets.iter_mut().zip(counts.iter()).skip(1)
                .filter_map(|(bucket, &count)| if count > 0 { Some(bucket) } else { None })
        });
        primary.into_iter().chain(tail.into_iter().flatten())
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
            let first_count = dist.count(0);
            for idx in 1..N {
                assert!(first_count >= dist.count(idx), "First count should be the largest");
            }
        }
        let num_seen = dist.num_observed();
        let kind = if dist.other() == 0 {
            // Seen <= N types total
            if dist.count(0) == 0 {
                DistributionKind::Empty
            } else if dist.count(1) == 0 {
                DistributionKind::Monomorphic
            } else if (dist.count(0) as f64)/(num_seen as f64) >= SKEW_THRESHOLD {
                DistributionKind::SkewedPolymorphic
            } else {
                DistributionKind::Polymorphic
            }
        } else {
            // Seen > N types total; considered megamorphic
            if (dist.count(0) as f64)/(num_seen as f64) >= SKEW_THRESHOLD {
                DistributionKind::SkewedMegamorphic
            } else {
                DistributionKind::Megamorphic
            }
        };
        Self { kind, buckets: dist.buckets() }
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
        assert_eq!(dist.other(), 0);
        assert!((0..4).all(|idx| dist.count(idx) == 0));
    }

    #[test]
    fn monomorphic_distribution_stays_unboxed() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        assert!(dist.tail.is_none());
    }

    #[test]
    fn observe_adds_record() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        assert_eq!(dist.bucket(0), 10);
        assert_eq!(dist.count(0), 1);
        assert_eq!(dist.other(), 0);
    }

    #[test]
    fn observe_increments_record() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        assert_eq!(dist.bucket(0), 10);
        assert_eq!(dist.count(0), 2);
        assert_eq!(dist.other(), 0);
    }

    #[test]
    fn observe_two() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        dist.observe(11);
        assert_eq!(dist.bucket(0), 11);
        assert_eq!(dist.count(0), 3);
        assert_eq!(dist.bucket(1), 10);
        assert_eq!(dist.count(1), 2);
        assert_eq!(dist.other(), 0);
    }

    #[test]
    fn observe_with_max_increments_other() {
        let mut dist = Distribution::<usize, 0>::new();
        dist.observe(10);
        assert_eq!(dist.num_buckets_used(), 0);
        assert_eq!(dist.other(), 1);
    }

    #[test]
    fn each_item_walks_buckets_in_order() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        dist.observe(10);
        dist.observe(11);
        assert_eq!(dist.each_item().collect::<Vec<_>>(), vec![10, 11]);
        for item in dist.each_item_mut() {
            *item += 100;
        }
        assert_eq!(dist.each_item().collect::<Vec<_>>(), vec![110, 111]);
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
