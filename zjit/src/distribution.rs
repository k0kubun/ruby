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
///
/// The wider the distribution, the better this trade: at `N` = 8 a monomorphic
/// `Distribution<ProfiledType, 8>` is 32 bytes instead of 144, and the seven
/// buckets a polymorphic site needs are paid for only by the ~2% of sites that
/// actually go polymorphic.
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

    /// Snapshot of every bucket's count, for [`DistributionSummary`].
    fn counts(&self) -> [NumProfiles; N] {
        let mut counts = [0; N];
        for (idx, count) in counts.iter_mut().enumerate() {
            *count = self.count(idx);
        }
        counts
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

    /// Look `item` up and, if `matches` finds no bucket holding it and a bucket is free, record
    /// it there. Unlike [`Self::observe`] this does not count the sighting and does not reorder
    /// the buckets, so the index it reports stays valid for as long as the distribution is only
    /// updated this way: a caller that remembers how many buckets were occupied at some earlier
    /// point can tell from the index alone whether an item was already known then.
    ///
    /// New buckets always start at a count of 1, which cannot exceed `count(0)`, so this
    /// preserves "bucket 0 is the most common item" without bubbling.
    ///
    /// Like [`Self::observe`], this only materializes the boxed tail once a second distinct
    /// item shows up, so an ivar site whose fallback keeps seeing the one shape it was
    /// compiled for never pays for buckets 1..N.
    pub fn observe_stable(&mut self, item: T, matches: impl Fn(T, T) -> bool) -> StableBucket {
        if N == 0 {
            self.other = self.other.saturating_add(1);
            return StableBucket::Full;
        }
        // Bucket 0 lives inline.
        if self.primary_count == 0 {
            self.primary = item;
            self.primary_count = 1;
            return StableBucket::Inserted(0);
        }
        if matches(self.primary, item) {
            return StableBucket::Existing(0);
        }
        // A second distinct item: materialize buckets 1..N. Reaching here means
        // `primary_count > 0`, so the "tail implies a non-empty bucket 0" invariant holds.
        let tail = self.tail.get_or_insert_with(|| Box::new(Tail::new()));
        for index in 1..N {
            if tail.counts[index] == 0 {
                tail.buckets[index] = item;
                tail.counts[index] = 1;
                return StableBucket::Inserted(index);
            }
            if matches(tail.buckets[index], item) {
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
        self.tail = None;
        self.other = 0;
    }

    /// Every item in a non-empty bucket, bucket 0 first.
    ///
    /// Mirrors [`Self::each_item_mut`] in reaching into the boxed tail once rather
    /// than asking `bucket`/`count` per index, which re-check `tail` every time. A
    /// monomorphic distribution has no tail at all, and the overwhelming majority
    /// are monomorphic, so this yields one item after one branch instead of walking
    /// `N` indices to find `N - 1` empty buckets. GC marking walks every
    /// distribution of every live profile on every major collection, which is where
    /// that difference shows up.
    pub fn each_item(&self) -> impl Iterator<Item = T> + '_ {
        let primary = if N > 0 && self.primary_count > 0 { Some(self.primary) } else { None };
        let tail = self.tail.as_ref().map(|tail| {
            tail.buckets.iter().zip(tail.counts.iter()).skip(1)
                .filter_map(|(&bucket, &count)| if count > 0 { Some(bucket) } else { None })
        });
        primary.into_iter().chain(tail.into_iter().flatten())
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

    /// Build a summary that claims too many types were seen to be worth specializing. Used for
    /// the fallthrough of a type-dispatch chain: every profiled type already has its own branch,
    /// so whatever reaches the fallthrough is by construction a type the profile never saw.
    pub fn megamorphic() -> Self {
        Self { kind: DistributionKind::Megamorphic, buckets: [Default::default(); N], counts: [0; N], other: NumProfiles::MAX }
    }

    /// Build a summary that looks like only `item` was ever observed. Used to hand a
    /// guarded arm of a polymorphic dispatch the one type it guarded on, and more
    /// generally to hand a single, already-narrowed observation to consumers that
    /// expect a distribution.
    pub fn monomorphic(item: T) -> Self {
        Self::monomorphic_variants(&[item])
    }

    /// Build a monomorphic summary out of several items that a consumer treats as one. Dispatch
    /// arms use this: the arm has already branched on the Ruby class, so every item in it is the
    /// same class as far as method lookup is concerned (hence `Monomorphic`), but the items still
    /// differ in the shape they carry, and a consumer that specializes on shape wants all of them.
    /// `items` beyond the bucket count are dropped, most significant first.
    pub fn monomorphic_variants(items: &[T]) -> Self {
        assert!(N > 0);
        assert!(!items.is_empty(), "a monomorphic summary needs at least one item");
        let mut buckets = [Default::default(); N];
        let mut counts = [0; N];
        for (i, &item) in items.iter().take(N).enumerate() {
            buckets[i] = item;
            counts[i] = 1;
        }
        Self { kind: DistributionKind::Monomorphic, buckets, counts, other: 0 }
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
        Self { kind, buckets: dist.buckets(), counts: dist.counts(), other: dist.other() }
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
    fn observe_stable_reports_new_and_known_buckets() {
        let mut dist = Distribution::<usize, 4>::new();
        dist.observe(10);
        // A bucket that already exists is reported without being counted again.
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Existing(0));
        assert_eq!(dist.count(0), 1);
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
        assert_eq!(dist.bucket(0), 10);
        assert_eq!(dist.bucket(1), 11);
        // Counts stay ordered, so DistributionSummary's invariant still holds.
        assert!(dist.count(0) >= dist.count(1));
    }

    #[test]
    fn observe_stable_reports_full_without_recording() {
        let mut dist = Distribution::<usize, 2>::new();
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Inserted(0));
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Inserted(1));
        assert_eq!(dist.observe_stable(12, |a, b| a == b), StableBucket::Full);
        assert_eq!(dist.other(), 1);
        assert_eq!(dist.buckets(), [10, 11]);
    }

    #[test]
    fn observe_stable_matches_with_the_given_predicate() {
        let mut dist = Distribution::<(usize, usize), 4>::new();
        dist.observe_stable((1, 100), |a, b| a.0 == b.0);
        // Equal under the predicate, different as a whole: still the same bucket.
        assert_eq!(dist.observe_stable((1, 999), |a, b| a.0 == b.0), StableBucket::Existing(0));
        assert_eq!(dist.bucket(0), (1, 100));
    }

    #[test]
    fn observe_stable_leaves_a_monomorphic_distribution_unboxed() {
        let mut dist = Distribution::<usize, 8>::new();
        // The common case for an ivar fallback: the shape it keeps seeing is the one bucket 0
        // already holds, so the boxed tail is never allocated.
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Inserted(0));
        for _ in 0..10 {
            assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Existing(0));
        }
        assert!(dist.tail.is_none());
        assert_eq!(dist.heap_size(), 0);
        // A second shape is what pays for the tail.
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Inserted(1));
        assert!(dist.tail.is_some());
    }

    #[test]
    fn observe_stable_and_observe_agree_on_bucket_zero() {
        // `observe_stable` may run against a distribution the interpreter filled with
        // `observe`, so bucket 0 has to mean the same thing to both.
        let mut dist = Distribution::<usize, 8>::new();
        dist.observe(10);
        dist.observe(11);
        dist.observe(11);
        // 11 bubbled to the front.
        assert_eq!(dist.bucket(0), 11);
        assert_eq!(dist.observe_stable(11, |a, b| a == b), StableBucket::Existing(0));
        assert_eq!(dist.observe_stable(10, |a, b| a == b), StableBucket::Existing(1));
        assert_eq!(dist.observe_stable(12, |a, b| a == b), StableBucket::Inserted(2));
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
