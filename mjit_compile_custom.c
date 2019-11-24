bool
mjit_compile_custom(FILE *f, const rb_iseq_t *iseq, struct compile_status *status)
{
    VALUE* opes = iseq->body->iseq_encoded;
    fprintf(f, "    VALUE stack[3];\n");
    fprintf(f, "    static const rb_iseq_t *original_iseq = (const rb_iseq_t *)0x%"PRIxVALUE";\n", (VALUE)iseq);
    fprintf(f, "    static const VALUE *const original_body_iseq = (VALUE *)0x%"PRIxVALUE";\n\n", (VALUE)iseq->body->iseq_encoded);

    fprintf(f, "label_0: /* getlocal_WC_0 */\n"); // k
    fprintf(f, "    stack[0] = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");

    fprintf(f, "label_2: /* opt_send_without_block */\n"); // downcase
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[3]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 4;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack[0];\n");
    fprintf(f, "    {\n");
    fprintf(f, "        VALUE bh = VM_BLOCK_HANDLER_NONE;\n");
    fprintf(f, "        val = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_method_wrap);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            return val;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_4: /* opt_send_without_block */\n"); // freeze
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[5]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 6;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack[0];\n");
    fprintf(f, "    {\n");
    fprintf(f, "        VALUE bh = VM_BLOCK_HANDLER_NONE;\n");
    fprintf(f, "        val = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_method_wrap);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            return val;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_6: /* setlocal_WC_0 */\n");
    fprintf(f, "    vm_env_write(vm_get_ep(GET_EP(), 0), -(int)0x3, stack[0]);\n");

    fprintf(f, "label_8: /* getinstancevariable */\n"); // @names
    fprintf(f, "{\n");
    fprintf(f, "    ID id = (ID)0x%"PRIxVALUE";\n", opes[9]);
    fprintf(f, "    IC ic = (IC)0x%"PRIxVALUE";\n", opes[10]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 11;\n");
    fprintf(f, "    stack[0] = vm_getinstancevariable(GET_SELF(), id, ic);\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_11: /* getlocal_WC_0 */\n"); // canonical
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x3);\n");

    fprintf(f, "label_13: /* opt_aref */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[14]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 15;\n");
    fprintf(f, "    {\n");
    fprintf(f, "        val = vm_opt_aref(stack[0], stack[1]);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "            reg_cfp->pc = original_body_iseq + 13;\n");
    fprintf(f, "            goto cancel;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_15: /* branchunless */\n");
    fprintf(f, "{\n");
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 17;\n");
    fprintf(f, "    if (!RTEST(stack[0])) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 0;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_17: /* getinstancevariable */\n"); // @names
    fprintf(f, "{\n");
    fprintf(f, "    MAYBE_UNUSED(VALUE) val;\n");
    fprintf(f, "    ID id = (ID)0x%"PRIxVALUE";\n", opes[18]);
    fprintf(f, "    IC ic = (IC)0x%"PRIxVALUE";\n", opes[19]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 20;\n");
    fprintf(f, "    stack[0] = vm_getinstancevariable(GET_SELF(), id, ic);\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_20: /* getlocal_WC_0 */\n"); // canonical
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x3);\n");

    fprintf(f, "label_22: /* opt_aref */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[23]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 24;\n");
    fprintf(f, "    val = vm_opt_aref(stack[0], stack[1]);\n");
    fprintf(f, "    if (val == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 22;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_24: /* getlocal_WC_0 */\n"); // k
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");

    fprintf(f, "label_26: /* opt_neq */\n");
    fprintf(f, "{\n");
    fprintf(f, "    MAYBE_UNUSED(CALL_DATA) cd, cd_eq;\n");
    fprintf(f, "    MAYBE_UNUSED(VALUE) obj, recv, val;\n");
    fprintf(f, "    cd_eq = (CALL_DATA)0x%"PRIxVALUE";\n", opes[27]);
    fprintf(f, "    cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[28]);
    fprintf(f, "    recv = stack[0];\n");
    fprintf(f, "    obj = stack[1];\n");
    fprintf(f, "    {\n");
    fprintf(f, "        val = vm_opt_neq(cd, cd_eq, recv, obj);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "            reg_cfp->pc = original_body_iseq + 26;\n");
    fprintf(f, "            goto cancel;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_29: /* branchunless */\n");
    fprintf(f, "{\n");
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 31;\n");
    fprintf(f, "    if (!RTEST(stack[0])) {\n");
    fprintf(f, "        RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "        goto label_37;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 0;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_31: /* putself */\n");
    fprintf(f, "    stack[0] = GET_SELF();\n");

    fprintf(f, "label_32: /* getlocal_WC_0 */\n"); // k
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");

    fprintf(f, "label_34: /* opt_send_without_block */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[35]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 36;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 2;\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = stack[0];\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack[1];\n");
    fprintf(f, "    {\n");
    fprintf(f, "        VALUE bh = VM_BLOCK_HANDLER_NONE;\n");
    fprintf(f, "        val = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_method_wrap);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            return val;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_37: /* getinstancevariable */\n"); // @names
    fprintf(f, "{\n");
    fprintf(f, "    ID id = (ID)0x%"PRIxVALUE";\n", opes[38]);
    fprintf(f, "    IC ic = (IC)0x%"PRIxVALUE";\n", opes[39]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 40;\n");
    fprintf(f, "    stack[0] = vm_getinstancevariable(GET_SELF(), id, ic);\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_40: /* getlocal_WC_0 */\n"); // canonical
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x3);\n");

    fprintf(f, "label_42: /* getlocal_WC_0 */\n"); // k
    fprintf(f, "    stack[2] = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");

    fprintf(f, "label_44: /* opt_aset */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[45]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 46;\n");
    fprintf(f, "    val = vm_opt_aset(stack[0], stack[1], stack[2]);\n");
    fprintf(f, "    if (val == Qundef) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 3;\n");
    fprintf(f, "        reg_cfp->pc = original_body_iseq + 44;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_47: /* putself */\n");
    fprintf(f, "    stack[0] = GET_SELF();\n");

    fprintf(f, "label_48: /* getlocal_WC_0 */\n"); // k
    fprintf(f, "    stack[1] = *(vm_get_ep(GET_EP(), 0) - 0x5);\n");

    fprintf(f, "label_50: /* getlocal_WC_0 */\n"); // v
    fprintf(f, "    stack[2] = *(vm_get_ep(GET_EP(), 0) - 0x4);\n");

    fprintf(f, "label_52: /* invokesuper */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val;\n");
    fprintf(f, "    CALL_DATA cd = (CALL_DATA)0x%"PRIxVALUE";\n", opes[53]);
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 55;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 3;\n");
    fprintf(f, "    *(reg_cfp->sp + -3) = stack[0];\n");
    fprintf(f, "    *(reg_cfp->sp + -2) = stack[1];\n");
    fprintf(f, "    *(reg_cfp->sp + -1) = stack[2];\n");
    fprintf(f, "    {\n");
    fprintf(f, "        VALUE bh = vm_caller_setup_arg_block(ec, GET_CFP(), &cd->ci, 0, true);\n");
    fprintf(f, "        val = vm_sendish(ec, GET_CFP(), cd, bh, vm_search_super_method);\n");
    fprintf(f, "        if (val == Qundef) {\n");
    fprintf(f, "            return val;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n");
    fprintf(f, "    stack[0] = val;\n");
    fprintf(f, "    if (UNLIKELY(!mjit_call_p)) {\n");
    fprintf(f, "        reg_cfp->sp = vm_base_ptr(reg_cfp) + 1;\n");
    fprintf(f, "        goto cancel;\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n\n");

    fprintf(f, "label_55: /* leave */\n");
    fprintf(f, "{\n");
    fprintf(f, "    VALUE val = stack[0];\n");
    fprintf(f, "    reg_cfp->pc = original_body_iseq + 56;\n");
    fprintf(f, "    reg_cfp->sp = vm_base_ptr(reg_cfp) + 0;\n");
    fprintf(f, "    RUBY_VM_CHECK_INTS(ec);\n");
    fprintf(f, "    if (vm_pop_frame(ec, GET_CFP(), GET_EP())) {\n");
    fprintf(f, "        return stack[0];\n");
    fprintf(f, "    }\n");
    fprintf(f, "    else {\n");
    fprintf(f, "        return stack[0];\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n");

    compile_cancel_handler(f, iseq->body, status);

    return true;
}
