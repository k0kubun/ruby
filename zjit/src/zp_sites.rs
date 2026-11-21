//! Instrumentation (not for landing): per-site counts of side exits and resume-entry decisions.
//! Enabled by ZP_SITE_EXITS=1. rb_zjit_zp_dump_sites(path) writes cumulative counts as TSV.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use crate::cruby::*;

pub struct Site {
    kind: &'static str,
    reason: String,
    loc: String,
    insn_idx: i64,
    compiled: String,
    version: usize,
    max_versions: usize,
    recompile: bool,
    lukewarm: bool,
    resume: bool,
    count: AtomicU64,
}

struct SitePtr(&'static Site);
unsafe impl Send for SitePtr {}

static SITES: Mutex<Vec<SitePtr>> = Mutex::new(Vec::new());
static RESUME_SITES: OnceLock<Mutex<HashMap<(usize, u16, u8, &'static str), &'static Site>>> = OnceLock::new();

/// The ISEQ version being compiled: (compiled ISEQ location, version index, max versions, lukewarm, resume)
pub static CURRENT: Mutex<Option<(String, usize, usize, bool, bool)>> = Mutex::new(None);

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ZP_SITE_EXITS").is_some())
}

pub fn set_current(iseq: IseqPtr, version: usize, max_versions: usize, lukewarm: bool, resume: bool) {
    if enabled() {
        uncounted(|| *CURRENT.lock().unwrap() = Some((iseq_get_location(iseq, 0), version, max_versions, lukewarm, resume)));
    }
}

fn insn_idx(iseq: IseqPtr, pc: *const VALUE) -> i64 {
    iseq_pc_to_insn_idx(iseq, pc as *mut VALUE).map_or(-1, |idx| idx as i64)
}

/// Register a side-exit site at compile time and return the pointer that the exit code passes to
/// rb_zjit_zp_site_hit().
pub fn new_exit_site(reason: &dyn std::fmt::Display, iseq: IseqPtr, pc: *const VALUE, recompile: bool) -> *const Site {
    uncounted(|| new_exit_site_body(reason.to_string(), iseq, pc, recompile))
}

/// Run `f` without counting its (leaked) allocations in zjit_alloc_bytes, so that the
/// instrumentation doesn't make ZJIT hit --zjit-mem-size sooner.
fn uncounted<R>(f: impl FnOnce() -> R) -> R {
    use std::sync::atomic::Ordering::SeqCst;
    let before = jit::GLOBAL_ALLOCATOR.alloc_size.load(SeqCst);
    let result = f();
    let after = jit::GLOBAL_ALLOCATOR.alloc_size.load(SeqCst);
    // Everything allocated or freed in `f` is the instrumentation's own memory, in both directions.
    if after > before {
        jit::GLOBAL_ALLOCATOR.alloc_size.fetch_sub(after - before, SeqCst);
    } else {
        jit::GLOBAL_ALLOCATOR.alloc_size.fetch_add(before - after, SeqCst);
    }
    result
}

fn new_exit_site_body(reason: String, iseq: IseqPtr, pc: *const VALUE, recompile: bool) -> *const Site {
    let (compiled, version, max_versions, lukewarm, resume) = CURRENT.lock().unwrap().clone().unwrap_or_default();
    let idx = insn_idx(iseq, pc);
    let site: &'static Site = Box::leak(Box::new(Site {
        kind: "exit",
        reason,
        loc: iseq_get_location(iseq, idx.max(0) as u32),
        insn_idx: idx,
        compiled,
        version,
        max_versions,
        recompile,
        lukewarm,
        resume,
        count: AtomicU64::new(0),
    }));
    SITES.lock().unwrap().push(SitePtr(site));
    site
}

#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_zp_site_hit(site: *const Site) {
    unsafe { &*site }.count.fetch_add(1, Ordering::Relaxed);
}

/// Count a resume-entry decision for a frame of `iseq` at `insn_idx`.
pub fn resume_hit(iseq: IseqPtr, insn_idx: u16, stack_size: u8, outcome: &'static str, num_versions: usize) {
    if !enabled() {
        return;
    }
    let map = RESUME_SITES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    let site = *map.entry((iseq as usize, insn_idx, stack_size, outcome)).or_insert_with(|| uncounted(|| {
        let site: &'static Site = Box::leak(Box::new(Site {
            kind: "resume",
            reason: outcome.to_string(),
            loc: iseq_get_location(iseq, insn_idx as u32),
            insn_idx: insn_idx as i64,
            compiled: format!("stack_size={stack_size}"),
            version: num_versions,
            max_versions: crate::codegen::max_iseq_versions(),
            recompile: false,
            lukewarm: false,
            resume: true,
            count: AtomicU64::new(0),
        }));
        SITES.lock().unwrap().push(SitePtr(site));
        site
    }));
    site.count.fetch_add(1, Ordering::Relaxed);
}

/// Write all sites with a non-zero count as TSV.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_zp_dump_sites(path: *const std::ffi::c_char) {
    let path = unsafe { std::ffi::CStr::from_ptr(path) }.to_string_lossy().into_owned();
    let Ok(mut file) = std::fs::File::create(&path) else { return };
    let _ = writeln!(file, "kind\treason\tloc\tinsn_idx\tcompiled\tversion\tmax_versions\trecompile\tlukewarm\tresume\tcount\tsite");
    for SitePtr(site) in SITES.lock().unwrap().iter() {
        let count = site.count.load(Ordering::Relaxed);
        if count > 0 {
            let _ = writeln!(file, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:p}", site.kind, site.reason, site.loc, site.insn_idx,
                site.compiled, site.version, site.max_versions, site.recompile, site.lukewarm, site.resume, count, *site);
        }
    }
}
