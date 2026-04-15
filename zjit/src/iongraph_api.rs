//! API for generating iongraph JSON from an ISEQ, used by the Wasm build
//! to visualize ZJIT's HIR without native code generation.

use crate::cruby::*;
use crate::hir::iseq_to_hir;
use crate::json::Json;

/// Build HIR from an ISEQ, run optimization passes, and return iongraph JSON as a Ruby string.
/// Called from Ruby as RubyVM::ZJIT.dump_iongraph(iseq).
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_dump_iongraph(iseq: IseqPtr) -> VALUE {
    // Catch panics to avoid UB for unwinding into C frames.
    let result = std::panic::catch_unwind(|| {
        dump_iongraph_inner(iseq)
    });

    match result {
        Ok(Some(val)) => val,
        Ok(None) => Qnil,
        Err(_) => Qnil,
    }
}

fn dump_iongraph_inner(iseq: IseqPtr) -> Option<VALUE> {
    // Build HIR from the ISEQ
    let mut function = match iseq_to_hir(iseq) {
        Ok(f) => f,
        Err(_) => return None,
    };

    // Run optimization passes and collect iongraph JSON for each
    let passes = function.optimize_into_iongraph();

    // Build the top-level JSON structure matching what iongraph.js expects
    let function_name = iseq_get_location(iseq, 0);
    let function_json = Json::object()
        .insert("name", function_name.as_str())
        .insert("passes", passes)
        .build();

    let result_json = Json::object()
        .insert("version", 1)
        .insert("functions", Json::array(vec![function_json]))
        .build();

    // Serialize to string
    let json_string = format!("{result_json}");

    // Convert to Ruby string
    Some(rust_str_to_ruby(&json_string))
}
