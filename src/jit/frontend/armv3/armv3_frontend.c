#include "jit/frontend/armv3/armv3_frontend.h"
#include "jit/frontend/armv3/armv3_disasm.h"
#include "jit/ir/ir.h"
#include "jit/jit.h"

static int armv3_frontend_analyze_code(struct jit_frontend *base, struct jit_block_meta *meta) {
  struct armv3_frontend *frontend = (struct armv3_frontend *)base;
  struct jit_guest *guest = frontend->jit->guest;

  meta->num_cycles = 0;
  meta->num_instrs = 0;
  meta->size = 0;

  while (1) {
    uint32_t data = guest->r32(guest->space, meta->guest_addr + meta->size);
    union armv3_instr i = {data};
    struct armv3_desc *desc = armv3_disasm(i.raw);

    /* end block on invalid instruction */
    if (desc->op == ARMV3_OP_INVALID) {
      return 0;
    }

    meta->num_cycles += 12;
    meta->num_instrs++;
    meta->size += 4;

    /* stop emitting when pc is changed */
    if ((desc->flags & FLAG_BRANCH) ||
        ((desc->flags & FLAG_DATA) && i.data.rd == 15) ||
        (desc->flags & FLAG_PSR) ||
        ((desc->flags & FLAG_XFR) && i.xfr.rd == 15) ||
        ((desc->flags & FLAG_BLK) && i.blk.rlist & (1 << 15)) ||
        (desc->flags & FLAG_SWI)) {
      break;
    }
  }

  return 1;
}

static void armv3_frontend_translate_code(struct jit_frontend *base, struct jit_code *code, struct ir *ir) {
  struct armv3_frontend *frontend = (struct armv3_frontend *)base;

  frontend->translate(frontend->data, code, ir);
}

static void armv3_frontend_dump_code(struct jit_frontend *base, uint32_t addr,
                                     int size) {
  struct armv3_frontend *frontend = (struct armv3_frontend *)base;
  struct jit_guest *guest = frontend->jit->guest;

  char buffer[128];

  for (int i = 0; i < size; i += 4) {
    uint32_t data = guest->r32(guest->space, addr);

    armv3_format(addr, data, buffer, sizeof(buffer));
    LOG_INFO(buffer);

    addr += 4;
  }
}

void armv3_frontend_destroy(struct armv3_frontend *frontend) {
  free(frontend);
}

struct armv3_frontend *armv3_frontend_create(struct jit *jit) {
  struct armv3_frontend *frontend = calloc(1, sizeof(struct armv3_frontend));

  frontend->jit = jit;
  frontend->analyze_code = &armv3_frontend_analyze_code;
  frontend->translate_code = &armv3_frontend_translate_code;
  frontend->dump_code = &armv3_frontend_dump_code;

  return frontend;
}
