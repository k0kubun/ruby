//! A fast, non-cryptographic hasher for maps and sets built by the compiler.

use std::hash::BuildHasherDefault;
use crate::cast::IntoU64;

/// The hash function of rustc's FxHasher: fast, but not resistant to HashDoS, which
/// doesn't matter for keys built by the compiler (instruction IDs, block IDs, operands).
/// SipHash, the default of std's HashMap, costs several times more per lookup.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) { self.add(i.into()) }
    #[inline]
    fn write_u16(&mut self, i: u16) { self.add(i.into()) }
    #[inline]
    fn write_u32(&mut self, i: u32) { self.add(i.into()) }
    #[inline]
    fn write_u64(&mut self, i: u64) { self.add(i) }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.add(i.as_u64()) }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
pub type HashSet<K> = std::collections::HashSet<K, FxBuildHasher>;
