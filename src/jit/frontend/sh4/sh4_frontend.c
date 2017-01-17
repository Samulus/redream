#include "jit/frontend/sh4/sh4_frontend.h"
#include "core/profiler.h"
#include "jit/frontend/sh4/sh4_context.h"
#include "jit/frontend/sh4/sh4_disasm.h"
#include "jit/frontend/sh4/sh4_translate.h"
#include "jit/ir/ir.h"
#include "jit/jit.h"

static int sh4_frontend_analyze_code(struct jit_frontend *base, struct jit_block_meta *meta) {
  struct sh4_frontend *frontend = (struct sh4_frontend *)base;
  struct jit_guest *guest = frontend->jit->guest;

  meta->num_cycles = 0;
  meta->num_instrs = 0;
  meta->size = 0;

  while (1) {
    struct sh4_instr instr = {0};
    instr.addr = meta->guest_addr + meta->size;
    instr.opcode = guest->r16(guest->space, instr.addr);

    /* end block on invalid instruction */
    if (!sh4_disasm(&instr)) {
      return 0;
    }

    meta->num_cycles += instr.cycles;
    meta->num_instrs++;
    meta->size += 2;

    if (instr.flags & SH4_FLAG_DELAYED) {
      struct sh4_instr delay_instr = {0};
      delay_instr.addr = meta->guest_addr + meta->size;
      delay_instr.opcode = guest->r16(guest->space, delay_instr.addr);

      CHECK(sh4_disasm(&delay_instr));
      CHECK(!(delay_instr.flags & SH4_FLAG_DELAYED));

      meta->num_cycles += delay_instr.cycles;
      meta->num_instrs++;
      meta->size += 2;
    }

    /* stop emitting once a branch is hit and save off branch information */
    if (instr.flags & SH4_FLAG_BRANCH) {
      if (instr.op == SH4_OP_BF) {
        uint32_t dest_addr = ((int8_t)instr.disp * 2) + instr.addr + 4;
        meta->branch_type = BRANCH_STATIC_FALSE;
        meta->branch_addr = dest_addr;
        meta->next_addr = instr.addr + 2;
      } else if (instr.op == SH4_OP_BFS) {
        uint32_t dest_addr = ((int8_t)instr.disp * 2) + instr.addr + 4;
        meta->branch_type = BRANCH_STATIC_FALSE;
        meta->branch_addr = dest_addr;
        meta->next_addr = instr.addr + 4;
      } else if (instr.op == SH4_OP_BT) {
        uint32_t dest_addr = ((int8_t)instr.disp * 2) + instr.addr + 4;
        meta->branch_type = BRANCH_STATIC_TRUE;
        meta->branch_addr = dest_addr;
        meta->next_addr = instr.addr + 2;
      } else if (instr.op == SH4_OP_BTS) {
        uint32_t dest_addr = ((int8_t)instr.disp * 2) + instr.addr + 4;
        meta->branch_type = BRANCH_STATIC_TRUE;
        meta->branch_addr = dest_addr;
        meta->next_addr = instr.addr + 4;
      } else if (instr.op == SH4_OP_BRA) {
        int32_t disp = ((instr.disp & 0xfff) << 20) >> 20;
        uint32_t dest_addr = (disp * 2) + instr.addr + 4;
        meta->branch_type = BRANCH_STATIC;
        meta->branch_addr = dest_addr;
      } else if (instr.op == SH4_OP_BRAF) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_BSR) {
        int32_t disp = ((instr.disp & 0xfff) << 20) >>
                   20; /* 12-bit displacement must be sign extended */
        uint32_t ret_addr = instr.addr + 4;
        uint32_t dest_addr = ret_addr + disp * 2;
        meta->branch_type = BRANCH_STATIC;
        meta->branch_addr = dest_addr;
      } else if (instr.op == SH4_OP_BSRF) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_JMP) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_JSR) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_RTS) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_RTE) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else if (instr.op == SH4_OP_TRAPA) {
        meta->branch_type = BRANCH_DYNAMIC;
      } else {
        LOG_FATAL("Unexpected branch op");
      }

      break;
    }

    /* if fpscr has changed, stop emitting since the fpu state is invalidated.
       also, if sr has changed, stop emitting as there are interrupts that
       possibly need to be handled */
    if (instr.flags & (SH4_FLAG_SET_FPSCR | SH4_FLAG_SET_SR)) {
      meta->branch_type = BRANCH_FALL_THROUGH;
      break;
    }

    /* used by debugger when stepping through instructions */
    /*if (meta->flags & SH4_SINGLE_INSTR) {
      break;
    }*/
  }

  return 1;
}

static void sh4_frontend_translate_code(struct jit_frontend *base, struct jit_code *code, struct ir *ir) {
  PROF_ENTER("cpu", "sh4_frontend_translate_code");

  struct sh4_frontend *frontend = (struct sh4_frontend *)base;
  frontend->translate(frontend->data, code, ir);

  PROF_LEAVE();
}

static void sh4_frontend_dump_code(struct jit_frontend *base, uint32_t addr,
                                   int size) {
  struct sh4_frontend *frontend = (struct sh4_frontend *)base;
  struct jit_guest *guest = frontend->jit->guest;

  char buffer[128];

  int i = 0;

  while (i < size) {
    struct sh4_instr instr = {0};
    instr.addr = addr + i;
    instr.opcode = guest->r16(guest->space, instr.addr);
    sh4_disasm(&instr);

    sh4_format(&instr, buffer, sizeof(buffer));
    LOG_INFO(buffer);

    i += 2;

    if (instr.flags & SH4_FLAG_DELAYED) {
      struct sh4_instr delay = {0};
      delay.addr = addr + i;
      delay.opcode = guest->r16(guest->space, delay.addr);
      sh4_disasm(&delay);

      sh4_format(&delay, buffer, sizeof(buffer));
      LOG_INFO(buffer);

      i += 2;
    }
  }
}

void sh4_frontend_destroy(struct sh4_frontend *frontend) {
  free(frontend);
}

struct sh4_frontend *sh4_frontend_create(struct jit *jit) {
  struct sh4_frontend *frontend = calloc(1, sizeof(struct sh4_frontend));

  frontend->jit = jit;
  frontend->analyze_code = &sh4_frontend_analyze_code;
  frontend->translate_code = &sh4_frontend_translate_code;
  frontend->dump_code = &sh4_frontend_dump_code;

  return frontend;
}
