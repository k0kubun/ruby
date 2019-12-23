bool
mjit_compile_custom(FILE *f, const rb_iseq_t *iseq, struct compile_status *status)
{
    VALUE* opes = iseq->body->iseq_encoded;
    IVC ic;
    fprintf(f, "    VALUE k = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");
    fprintf(f, "    static const rb_iseq_t *original_iseq = (const rb_iseq_t *)0x%"PRIxVALUE";\n", (VALUE)iseq);
    fprintf(f, "    static const VALUE *const original_body_iseq = (VALUE *)0x%"PRIxVALUE";\n\n", (VALUE)iseq->body->iseq_encoded);

    // label_0: getlocal_WC_0 - k
    // label_2: opt_send_without_block - #downcase
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = k;\n");
    fprintf(f, "    VALUE stack_0 = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[3]);

    // label_4: opt_send_without_block - #freeze
    // label_6: setlocal - canonical
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack_0;\n");
    fprintf(f, "    VALUE canonical = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[5]);

    // label_8: getinstancevariable - @names
    ic = (IVC)opes[10];
    fprintf(f, "{\n");
    fprintf(f, "    VALUE obj = GET_SELF();\n");
    fprintf(f, "    const rb_serial_t ic_serial = (rb_serial_t)%"PRI_SERIALT_PREFIX"u;\n", ic->ic_serial);
    fprintf(f, "    const st_index_t index = %"PRIuSIZE";\n", ic->index);
    fprintf(f, "    struct gen_ivtbl *ivtbl;\n");
    fprintf(f, "    st_lookup(rb_ivar_generic_ivtbl(), (st_data_t)obj, (st_data_t *)&ivtbl); stack_0 = ivtbl->ivptr[index];\n");
    fprintf(f, "}\n\n");

    // label_11: getlocal_WC_0 - canonical
    // label_13: opt_aref
    fprintf(f, "    stack_0 = vm_opt_aref(stack_0, canonical);\n");

    // label_15: branchunless
    fprintf(f, "    if (!RTEST(stack_0)) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");

    // label_17: getinstancevariable - @names
    ic = (IVC)opes[19];
    fprintf(f, "{\n");
    fprintf(f, "    VALUE obj = GET_SELF();\n");
    fprintf(f, "    const rb_serial_t ic_serial = (rb_serial_t)%"PRI_SERIALT_PREFIX"u;\n", ic->ic_serial);
    fprintf(f, "    const st_index_t index = %"PRIuSIZE";\n", ic->index);
    fprintf(f, "    struct gen_ivtbl *ivtbl;\n");
    fprintf(f, "    st_lookup(rb_ivar_generic_ivtbl(), (st_data_t)obj, (st_data_t *)&ivtbl); stack_0 = ivtbl->ivptr[index];\n");
    fprintf(f, "}\n\n");

    // label_20: getlocal_WC_0 - canonical
    // label_22: opt_aref
    fprintf(f, "    stack_0 = vm_opt_aref(stack_0, canonical);\n");

    // label_24: getlocal_WC_0 - k
    // label_26: opt_neq
    fprintf(f, "    stack_0 = vm_opt_neq((CALL_DATA)0x%"PRIxVALUE", (CALL_DATA)0x%"PRIxVALUE", stack_0, k);\n", opes[28], opes[27]);

    // label_29: branchunless
    fprintf(f, "    if (!RTEST(stack_0)) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");

    // label_31: putself
    // label_32: getlocal_WC_0 - k
    // label_34: opt_send_without_block
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = GET_SELF();\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = k;\n");
    fprintf(f, "    stack_0 = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[35]);

    fprintf(f, "label_37: /* getinstancevariable */\n"); // @names
    ic = (IVC)opes[39];
    fprintf(f, "{\n");
    fprintf(f, "    VALUE obj = GET_SELF();\n");
    fprintf(f, "    const rb_serial_t ic_serial = (rb_serial_t)%"PRI_SERIALT_PREFIX"u;\n", ic->ic_serial);
    fprintf(f, "    const st_index_t index = %"PRIuSIZE";\n", ic->index);
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    struct gen_ivtbl *ivtbl;\n");
    fprintf(f, "    st_lookup(rb_ivar_generic_ivtbl(), (st_data_t)obj, (st_data_t *)&ivtbl); stack_0 = ivtbl->ivptr[index];\n");
    fprintf(f, "}\n\n");

    // label_40: getlocal_WC_0 - canonical
    // label_42: getlocal_WC_0 - k
    // label_44: opt_aset
    fprintf(f, "    vm_opt_aset(stack_0, canonical, k);\n");

    // label_47: putself
    // label_48: getlocal_WC_0 - k
    // label_50: getlocal_WC_0 - v
    // label_52: invokesuper
    fprintf(f, "{\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[53]);
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 3;\n");
    fprintf(f, "    *(reg_cfp->sp + -3) = GET_SELF();\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = k;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = *(vm_get_ep(GET_EP(), 0) - 0x4);\n"); // v
    fprintf(f, "    VALUE bh = vm_caller_setup_arg_block(ec, GET_CFP(), &cd->ci, 0, true);\n");
    fprintf(f, "    stack_0 = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_super_method);\n");
    fprintf(f, "}\n\n");

    // label_55: leave
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 0;\n");
    fprintf(f, "    RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "    vm_pop_frame(ec, GET_CFP(), GET_EP());\n");
    fprintf(f, "    return stack_0;\n");

    compile_cancel_handler(f, iseq->body, status);

    return true;
}
