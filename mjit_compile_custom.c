bool
mjit_compile_custom(FILE *f, const rb_iseq_t *iseq, struct compile_status *status)
{
    VALUE* opes = iseq->body->iseq_encoded;
    fprintf(f, "    VALUE k = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");
    fprintf(f, "    VALUE stack_0, stack_1, stack_2;\n");
    fprintf(f, "    static const rb_iseq_t *original_iseq = (const rb_iseq_t *)0x%"PRIxVALUE";\n", (VALUE)iseq);
    fprintf(f, "    static const VALUE *const original_body_iseq = (VALUE *)0x%"PRIxVALUE";\n\n", (VALUE)iseq->body->iseq_encoded);

    // label_0: getlocal_WC_0 - k
    fprintf(f, "    stack_0 = k;\n");

    // label_2: opt_send_without_block - #downcase
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 4;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack_0;\n");
    fprintf(f, "    stack_0 = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[3]);
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        return Qundef;\n");
    fprintf(f, "    }\n");

    // label_4: opt_send_without_block - #freeze
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 6;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack_0;\n");
    fprintf(f, "    stack_0 = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[5]);
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        return Qundef;\n");
    fprintf(f, "    }\n");

    // label_6: setlocal - canonical
    fprintf(f, "    VALUE canonical = stack_0;\n");

    // label_8: getinstancevariable - @names
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 11;\n");
    fprintf(f, "    stack_0 = vm_getinstancevariable(GET_SELF(), (ID)0x%"PRIxVALUE", (IC)0x%"PRIxVALUE");\n", opes[9], opes[10]);

    // label_11: getlocal_WC_0 - canonical
    fprintf(f, "    stack_1 = canonical;\n");

    // label_13: opt_aref
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 15;\n");
    fprintf(f, "    stack_0 = vm_opt_aref(stack_0, stack_1);\n");
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 13;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");

    // label_15: branchunless
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 17;\n");
    fprintf(f, "    if (!RTEST(stack_0)) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");

    // label_17: getinstancevariable - @names
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 20;\n");
    fprintf(f, "    stack_0 = vm_getinstancevariable(GET_SELF(), (ID)0x%"PRIxVALUE", (IC)0x%"PRIxVALUE");\n", opes[18], opes[19]);

    // label_20: getlocal_WC_0 - canonical
    fprintf(f, "    stack_1 = canonical;\n");

    // label_22: opt_aref
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 24;\n");
    fprintf(f, "    VALUE val = vm_opt_aref(stack_0, stack_1);\n");
    fprintf(f, "    if (val == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 22;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack_0 = val;\n");

    // label_24: getlocal_WC_0 - k
    fprintf(f, "    stack_1 = k;\n");

    // label_26: opt_neq
    fprintf(f, "    stack_0 = vm_opt_neq((CALL_DATA)0x%"PRIxVALUE", (CALL_DATA)0x%"PRIxVALUE", stack_0, stack_1);\n", opes[28], opes[27]);
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 26;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");

    // label_29: branchunless
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 31;\n");
    fprintf(f, "    if (!RTEST(stack_0)) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");

    // label_31: putself
    fprintf(f, "    stack_0 = GET_SELF();\n");

    // label_32: getlocal_WC_0 - k
    fprintf(f, "    stack_1 = k;\n");

    // label_34: opt_send_without_block
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 36;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = stack_0;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack_1;\n");
    fprintf(f, "    stack_0 = vm_sendish(ec, GET_CFP(), (CALL_DATA)0x%"PRIxVALUE", VM_BLOCK_HANDLER_NONE, vm_search_method_wrap);\n", opes[35]);
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        return Qundef;\n");
    fprintf(f, "    }\n");

    fprintf(f, "label_37: /* getinstancevariable */\n"); // @names
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 40;\n");
    fprintf(f, "    stack_0 = vm_getinstancevariable(GET_SELF(), (ID)0x%"PRIxVALUE", (IC)0x%"PRIxVALUE");\n", opes[38], opes[39]);

    // label_40: getlocal_WC_0 - canonical
    fprintf(f, "    stack_1 = canonical;\n");

    // label_42: getlocal_WC_0 - k
    fprintf(f, "    stack_2 = k;\n");

    // label_44: opt_aset
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 46;\n");
    fprintf(f, "    stack_0 = vm_opt_aset(stack_0, stack_1, stack_2);\n");
    fprintf(f, "    if (stack_0 == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 3;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 44;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");

    // label_47: putself
    fprintf(f, "    stack_0 = GET_SELF();\n");

    // label_48: getlocal_WC_0 - k
    fprintf(f, "    stack_1 = k;\n");

    // label_50: getlocal_WC_0 - v
    fprintf(f, "    stack_2 = *(vm_get_ep(GET_EP(), 0) - 0x4);\n");

    // label_52: invokesuper
    fprintf(f, "{\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[53]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 55;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 3;\n");
    fprintf(f, "    *(reg_cfp->sp + -3) = stack_0;\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = stack_1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack_2;\n");
    fprintf(f, "    VALUE bh = vm_caller_setup_arg_block(ec, GET_CFP(), &cd->ci, 0, true);\n");
    fprintf(f, "    VALUE val = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_super_method);\n");
    fprintf(f, "    if (val == Qundef) {\n");
    fprintf(f, "        return val;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack_0 = val;\n");
    fprintf(f, "}\n\n");

    // label_55: leave
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 56;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 0;\n");
    fprintf(f, "    RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "    vm_pop_frame(ec, GET_CFP(), GET_EP());\n");
    fprintf(f, "    return stack_0;\n");

    compile_cancel_handler(f, iseq->body, status);

    return true;
}
