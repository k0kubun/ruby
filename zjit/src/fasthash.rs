//! A fast, non-cryptographic hasher for the compiler's internal hash tables.
//!
//! `std`'s default hasher is SipHash-1-3, which is the right default for a table whose
//! keys can come from an attacker. None of ZJIT's compile-time tables are like that:
//! their keys are instruction operands, block ids and interpreter pointers that the
//! compiler produced itself. Paying SipHash for them is pure overhead, and it is not
//! small -- deduplicating side exits hashes every operand of every exit's stack and
//! locals, which on a compile-heavy workload was several percent of the whole process.
//!
//! This is the FxHash construction (multiply-xor-rotate) used by rustc for the same
//! reason. It is a good hash for the small integers and pointers we feed it and it
//! compiles down to a couple of instructions per word.

use std::hash::{BuildHasherDefault, Hasher};

/// Drop-in replacement for [`std::collections::HashMap`] with [`FastHasher`].
pub type FastHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;
/// Drop-in replacement for [`std::collections::HashSet`] with [`FastHasher`].
pub type FastHashSet<T> = std::collections::HashSet<T, BuildHasherDefault<FastHasher>>;

/// Odd 64-bit constant close to 2^64 / phi, so that multiplying by it spreads the
/// input bits across the whole word.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

#[derive(Default, Clone, Copy)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add_word(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Whole words first, then whatever is left over, so that a byte slice hashes in
        // len/8 rounds instead of len.
        let mut rest = bytes;
        while let Some((chunk, tail)) = rest.split_first_chunk::<8>() {
            self.add_word(u64::from_ne_bytes(*chunk));
            rest = tail;
        }
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add_word(u64::from_ne_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) { self.add_word(value as u64); }
    #[inline]
    fn write_u16(&mut self, value: u16) { self.add_word(value as u64); }
    #[inline]
    fn write_u32(&mut self, value: u32) { self.add_word(value as u64); }
    #[inline]
    fn write_u64(&mut self, value: u64) { self.add_word(value); }
    #[inline]
    fn write_usize(&mut self, value: usize) { self.add_word(value as u64); }
    #[inline]
    fn write_i8(&mut self, value: i8) { self.add_word(value as u64); }
    #[inline]
    fn write_i16(&mut self, value: i16) { self.add_word(value as u64); }
    #[inline]
    fn write_i32(&mut self, value: i32) { self.add_word(value as u64); }
    #[inline]
    fn write_i64(&mut self, value: i64) { self.add_word(value as u64); }
    #[inline]
    fn write_isize(&mut self, value: isize) { self.add_word(value as u64); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = FastHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn distinguishes_similar_keys() {
        // Sanity: the hash has to actually depend on every part of the key, including
        // element order, or the tables built on it degrade into linked lists.
        assert_ne!(hash_of(&(1u64, 2u64)), hash_of(&(2u64, 1u64)));
        assert_ne!(hash_of(&vec![1u32, 2, 3]), hash_of(&vec![1u32, 3, 2]));
        assert_ne!(hash_of(&0u64), hash_of(&1u64));
        assert_ne!(hash_of(&"abc"), hash_of(&"abd"));
    }

    #[test]
    fn works_as_a_hashmap_hasher() {
        let mut map: FastHashMap<(u64, u64), u64> = FastHashMap::default();
        for i in 0..1000u64 {
            map.insert((i, i * 7), i);
        }
        assert_eq!(map.len(), 1000);
        for i in 0..1000u64 {
            assert_eq!(map.get(&(i, i * 7)), Some(&i));
        }
        assert_eq!(map.get(&(1000, 7000)), None);
    }
}
