//! Optimized bitset implementation.

type Entry = u128;

const ENTRY_NUM_BITS: usize = Entry::BITS as usize;

// TODO(max): Make a `SmallBitSet` and `LargeBitSet` and switch between them if `num_bits` fits in
// `Entry`.
#[derive(Clone)]
pub struct BitSet<T: Into<usize> + Copy> {
    entries: Vec<Entry>,
    num_bits: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Into<usize> + Copy> BitSet<T> {
    pub fn with_capacity(num_bits: usize) -> Self {
        let num_entries = num_bits.div_ceil(ENTRY_NUM_BITS);
        Self { entries: vec![0; num_entries], num_bits, phantom: Default::default() }
    }

    /// Returns whether the value was newly inserted: true if the set did not originally contain
    /// the bit, and false otherwise.
    pub fn insert(&mut self, idx: T) -> bool {
        debug_assert!(idx.into() < self.num_bits);
        let entry_idx = idx.into() / ENTRY_NUM_BITS;
        let bit_idx = idx.into() % ENTRY_NUM_BITS;
        let newly_inserted = (self.entries[entry_idx] & (1 << bit_idx)) == 0;
        self.entries[entry_idx] |= 1 << bit_idx;
        newly_inserted
    }

    /// Set all bits to 0, keeping the allocation. Lets a dataflow loop reuse one scratch
    /// set instead of allocating a fresh one per block per iteration.
    pub fn clear(&mut self) {
        self.entries.fill(0);
    }

    /// Overwrite `self` with `other`, keeping `self`'s allocation.
    /// `self` and `other` must have the same number of bits.
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.num_bits, other.num_bits);
        self.entries.copy_from_slice(&other.entries);
    }

    /// Set all bits to 1.
    pub fn insert_all(&mut self) {
        for i in 0..self.entries.len() {
            self.entries[i] = !0;
        }
    }

    /// Clear a bit. Returns whether the bit was previously set.
    pub fn remove(&mut self, idx: T) -> bool {
        debug_assert!(idx.into() < self.num_bits);
        let entry_idx = idx.into() / ENTRY_NUM_BITS;
        let bit_idx = idx.into() % ENTRY_NUM_BITS;
        let was_set = (self.entries[entry_idx] & (1 << bit_idx)) != 0;
        self.entries[entry_idx] &= !(1 << bit_idx);
        was_set
    }

    pub fn get(&self, idx: T) -> bool {
        debug_assert!(idx.into() < self.num_bits);
        let entry_idx = idx.into() / ENTRY_NUM_BITS;
        let bit_idx = idx.into() % ENTRY_NUM_BITS;
        (self.entries[entry_idx] & (1 << bit_idx)) != 0
    }

    /// Modify `self` to only have bits set if they are also set in `other`. Returns true if `self`
    /// was modified, and false otherwise.
    /// `self` and `other` must have the same number of bits.
    pub fn intersect_with(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let mut changed = false;
        for i in 0..self.entries.len() {
            let before = self.entries[i];
            self.entries[i] &= other.entries[i];
            changed |= self.entries[i] != before;
        }
        changed
    }

    /// Modify `self` to have bits set if they are set in either `self` or `other`. Returns true if `self`
    /// was modified, and false otherwise.
    /// `self` and `other` must have the same number of bits.
    pub fn union_with(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let mut changed = false;
        for i in 0..self.entries.len() {
            let before = self.entries[i];
            self.entries[i] |= other.entries[i];
            changed |= self.entries[i] != before;
        }
        changed
    }

    /// Modify `self` to remove bits that are set in `other`. Returns true if `self`
    /// was modified, and false otherwise.
    /// `self` and `other` must have the same number of bits.
    pub fn difference_with(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let mut changed = false;
        for i in 0..self.entries.len() {
            let before = self.entries[i];
            self.entries[i] &= !other.entries[i];
            changed |= self.entries[i] != before;
        }
        changed
    }

    /// Check if two BitSets are equal.
    /// `self` and `other` must have the same number of bits.
    pub fn equals(&self, other: &Self) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        self.entries == other.entries
    }

    /// Returns an iterator over the indices of set bits.
    /// Only iterates over bits that are set, not all possible indices.
    pub fn iter_set_bits(&self) -> impl Iterator<Item = usize> + '_ {
        iter_set_bits(&self.entries, self.num_bits)
    }

    /// Union in one row of a [`BitMatrix`]. Returns whether `self` changed.
    pub fn union_with_row(&mut self, other: BitRow<'_, T>) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let mut changed = false;
        for (entry, &other_entry) in self.entries.iter_mut().zip(other.entries) {
            let before = *entry;
            *entry |= other_entry;
            changed |= *entry != before;
        }
        changed
    }

    /// Remove the bits set in one row of a [`BitMatrix`]. Returns whether `self` changed.
    pub fn difference_with_row(&mut self, other: BitRow<'_, T>) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let mut changed = false;
        for (entry, &other_entry) in self.entries.iter_mut().zip(other.entries) {
            let before = *entry;
            *entry &= !other_entry;
            changed |= *entry != before;
        }
        changed
    }

    /// Overwrite `self` with one row of a [`BitMatrix`], keeping `self`'s allocation.
    pub fn copy_from_row(&mut self, other: BitRow<'_, T>) {
        debug_assert_eq!(self.num_bits, other.num_bits);
        self.entries.copy_from_slice(other.entries);
    }

    /// Whether `self` holds exactly the bits in one row of a [`BitMatrix`].
    pub fn equals_row(&self, other: BitRow<'_, T>) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        self.entries == other.entries
    }
}

/// Iterate the indices of the set bits in a row of entries.
fn iter_set_bits(entries: &[Entry], num_bits: usize) -> impl Iterator<Item = usize> + '_ {
    entries.iter().enumerate().flat_map(move |(entry_idx, &entry)| {
        let mut bits = entry;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let bit_pos = bits.trailing_zeros() as usize;
            bits &= bits - 1; // Clear the lowest set bit
            Some(entry_idx * ENTRY_NUM_BITS + bit_pos)
        })
    }).filter(move |&idx| idx < num_bits)
}

/// One row of a [`BitMatrix`], borrowed. Same bit-set semantics as [`BitSet`],
/// but with no allocation of its own.
#[derive(Clone, Copy)]
pub struct BitRow<'a, T: Into<usize> + Copy> {
    entries: &'a [Entry],
    num_bits: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Into<usize> + Copy> BitRow<'a, T> {
    pub fn get(&self, idx: T) -> bool {
        debug_assert!(idx.into() < self.num_bits);
        let entry_idx = idx.into() / ENTRY_NUM_BITS;
        let bit_idx = idx.into() % ENTRY_NUM_BITS;
        (self.entries[entry_idx] & (1 << bit_idx)) != 0
    }

    pub fn iter_set_bits(&self) -> impl Iterator<Item = usize> + 'a {
        iter_set_bits(self.entries, self.num_bits)
    }
}

/// `rows` bit sets of `num_bits` bits each, in a single allocation.
///
/// Dataflow passes keep one set per basic block, and the obvious
/// `vec![BitSet::with_capacity(num_vregs); num_blocks]` spelling costs a heap
/// allocation *and* a memcpy per block per set: liveness analysis alone built
/// three of those per compile, which measured as the largest single source of
/// allocator traffic in the backend. The row width is fixed at construction,
/// which is all the dataflow passes need.
pub struct BitMatrix<T: Into<usize> + Copy> {
    entries: Vec<Entry>,
    /// Entries per row. Rows are laid out contiguously.
    row_entries: usize,
    num_bits: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Into<usize> + Copy> BitMatrix<T> {
    /// A matrix with every bit clear.
    pub fn new(rows: usize, num_bits: usize) -> Self {
        let row_entries = num_bits.div_ceil(ENTRY_NUM_BITS);
        Self {
            entries: vec![0; rows * row_entries],
            row_entries,
            num_bits,
            phantom: Default::default(),
        }
    }

    fn range(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * self.row_entries;
        start..start + self.row_entries
    }

    /// Borrow one row.
    pub fn row(&self, row: usize) -> BitRow<'_, T> {
        BitRow {
            entries: &self.entries[self.range(row)],
            num_bits: self.num_bits,
            phantom: Default::default(),
        }
    }

    /// Returns whether the bit was newly inserted.
    pub fn insert(&mut self, row: usize, idx: T) -> bool {
        debug_assert!(idx.into() < self.num_bits);
        let entry_idx = row * self.row_entries + idx.into() / ENTRY_NUM_BITS;
        let bit = 1 << (idx.into() % ENTRY_NUM_BITS);
        let newly_inserted = (self.entries[entry_idx] & bit) == 0;
        self.entries[entry_idx] |= bit;
        newly_inserted
    }

    pub fn get(&self, row: usize, idx: T) -> bool {
        self.row(row).get(idx)
    }

    /// Set every bit in one row.
    pub fn insert_all_row(&mut self, row: usize) {
        let range = self.range(row);
        self.entries[range].fill(!0);
    }

    /// Intersect one row with `other`, returning whether the row changed.
    pub fn intersect_row_with(&mut self, row: usize, other: &BitSet<T>) -> bool {
        debug_assert_eq!(self.num_bits, other.num_bits);
        let range = self.range(row);
        let mut changed = false;
        for (entry, &other_entry) in self.entries[range].iter_mut().zip(&other.entries) {
            let before = *entry;
            *entry &= other_entry;
            changed |= *entry != before;
        }
        changed
    }

    /// Overwrite one row with the contents of `src`, which must be the same width.
    pub fn copy_row_from(&mut self, row: usize, src: &BitSet<T>) {
        debug_assert_eq!(self.num_bits, src.num_bits);
        let range = self.range(row);
        self.entries[range].copy_from_slice(&src.entries);
    }
}

#[cfg(test)]
mod tests {
    use super::BitSet;

    #[test]
    #[should_panic]
    fn get_over_capacity_panics() {
        let set = BitSet::with_capacity(0);
        assert!(!set.get(0usize));
    }

    #[test]
    fn with_capacity_defaults_to_zero() {
        let set = BitSet::with_capacity(4);
        assert!(!set.get(0usize));
        assert!(!set.get(1usize));
        assert!(!set.get(2usize));
        assert!(!set.get(3usize));
    }

    #[test]
    fn insert_sets_bit() {
        let mut set = BitSet::with_capacity(4);
        assert!(set.insert(1usize));
        assert!(set.get(1usize));
    }

    #[test]
    fn insert_with_set_bit_returns_false() {
        let mut set = BitSet::with_capacity(4);
        assert!(set.insert(1usize));
        assert!(!set.insert(1usize));
    }

    #[test]
    fn insert_all_sets_all_bits() {
        let mut set = BitSet::with_capacity(4);
        set.insert_all();
        assert!(set.get(0usize));
        assert!(set.get(1usize));
        assert!(set.get(2usize));
        assert!(set.get(3usize));
    }

    #[test]
    #[should_panic]
    fn intersect_with_panics_with_different_num_bits() {
        let mut left: BitSet<usize> = BitSet::with_capacity(3);
        let right = BitSet::with_capacity(4);
        left.intersect_with(&right);
    }
    #[test]
    fn intersect_with_keeps_only_common_bits() {
        let mut left = BitSet::with_capacity(3);
        let mut right = BitSet::with_capacity(3);
        left.insert(0usize);
        left.insert(1usize);
        right.insert(1usize);
        right.insert(2usize);
        left.intersect_with(&right);
        assert!(!left.get(0usize));
        assert!(left.get(1usize));
        assert!(!left.get(2usize));
    }

    #[test]
    fn test_iter_set_bits() {
        let mut set: BitSet<usize> = BitSet::with_capacity(10);
        set.insert(1usize);
        set.insert(5usize);
        set.insert(9usize);

        let set_bits: Vec<usize> = set.iter_set_bits().collect();
        assert_eq!(set_bits, vec![1, 5, 9]);
    }

    #[test]
    fn test_iter_set_bits_empty() {
        let set: BitSet<usize> = BitSet::with_capacity(10);
        let set_bits: Vec<usize> = set.iter_set_bits().collect();
        assert_eq!(set_bits, vec![]);
    }

    #[test]
    fn test_iter_set_bits_all() {
        let mut set: BitSet<usize> = BitSet::with_capacity(5);
        set.insert_all();
        let set_bits: Vec<usize> = set.iter_set_bits().collect();
        assert_eq!(set_bits, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_iter_set_bits_large() {
        let mut set: BitSet<usize> = BitSet::with_capacity(200);
        set.insert(0usize);
        set.insert(127usize);
        set.insert(128usize);
        set.insert(199usize);

        let set_bits: Vec<usize> = set.iter_set_bits().collect();
        assert_eq!(set_bits, vec![0, 127, 128, 199]);
    }
}
