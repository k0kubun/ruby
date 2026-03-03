use crate::cruby::VALUE;

#[derive(Debug)]
pub struct JITFrame {
    pub pc: *const VALUE,
}

#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_jit_return_pc(jit_return: *const JITFrame) -> *const VALUE {
    unsafe { (*jit_return).pc }
}

#[cfg(test)]
mod tests {
    use crate::cruby::{eval, inspect};
    use insta::assert_snapshot;

    #[test]
    fn test_jit_frame_entry_first() {
        eval(r#"
            def test
              itself
              callee
            end

            def callee
              caller
            end

            test
        "#);
        assert_snapshot!(inspect("test.first"), @r#""<compiled>:4:in 'Object#test'""#);
    }

    #[test]
    fn test_materialize_one_frame() {
        assert_snapshot!(inspect("
            def jit_entry
              raise rescue 1
            end
            jit_entry
            jit_entry
        "), @"1");
    }

    #[test]
    fn test_materialize_two_frames() {
        // At the point of `resuce`, there are two lightweight frames on stack and both need to be
        // materialized before passing control to interpreter.
        assert_snapshot!(inspect("
            def jit_entry = raise_and_rescue
            def raise_and_rescue
              raise rescue 1
            end
            jit_entry
            jit_entry
        "), @"1");
    }
}
