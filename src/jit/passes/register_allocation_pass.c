#include <limits.h>
#include "jit/passes/register_allocation_pass.h"
#include "core/list.h"
#include "core/math.h"
#include "jit/backend/jit_backend.h"
#include "jit/ir/ir.h"
#include "jit/pass_stats.h"

/* second-chance binpacking register allocator based off of the paper "Quality
   and Speed in Linear-scan Register Allocation" by Omri Traub, Glenn Holloway
   and Michael D. Smith */

DEFINE_STAT(gprs_spilled, "gprs spilled");
DEFINE_STAT(fprs_spilled, "fprs spilled");

struct tmp;

struct bin {
  const struct jit_register *reg;

  /* current temporary */
  struct tmp *tmp;
};

/* allocation candidate */
struct tmp {
  int *uses;
  int num_uses;
  int max_uses;

  /* next use */
  int next;
  /* current register */
  struct ir_value *value;
  /* current stack spill slot */
  struct ir_local *slot;
};

struct ra {
  /* backend register information */
  const struct jit_register *regs;
  int num_regs;

  struct bin *bins;

  struct tmp *tmps;
  int num_tmps;
  int max_tmps;
};

#define ra_ordinal(i) ((int)(i)->tag)
#define ra_set_ordinal(i, ord) (i)->tag = (intptr_t)(ord)

#define ra_tmp(v) (&ra->tmps[(v)->tag])
#define ra_set_tmp(v, tmp) (v)->tag = (tmp)-ra->tmps

static int ra_reg_can_store(const struct jit_register *reg,
                            const struct ir_value *v) {
  int mask = 1 << v->type;
  return (reg->value_types & mask) == mask;
}

static void ra_add_use(struct tmp *tmp, int ordinal) {
  if (tmp->num_uses >= tmp->max_uses) {
    /* grow array */
    tmp->max_uses = MAX(32, tmp->max_uses * 2);
    tmp->uses = realloc(tmp->uses, tmp->max_uses * sizeof(int));
  }

  tmp->uses[tmp->num_uses++] = ordinal;
}

static struct tmp *ra_create_tmp(struct ra *ra, struct ir_value *value) {
  if (ra->num_tmps >= ra->max_tmps) {
    /* grow array */
    int old_max = ra->max_tmps;
    ra->max_tmps = MAX(32, ra->max_tmps * 2);
    ra->tmps = realloc(ra->tmps, ra->max_tmps * sizeof(struct tmp));

    /* initialize the new entries */
    memset(ra->tmps + old_max, 0, (ra->max_tmps - old_max) * sizeof(struct tmp));
  }

  /* reset the temporary's state, reusing the previously allocated uses array */
  struct tmp *tmp = &ra->tmps[ra->num_tmps];
  tmp->num_uses = 0;
  tmp->next = 0;
  tmp->value = NULL;
  tmp->slot = NULL;

  /* assign the temporary to the value */
  value->tag = ra->num_tmps++;

  return tmp;
}

static void ra_validate_r(struct ra *ra, struct ir *ir, struct ir_block *block,
                          struct ir_value **active_in) {
  size_t active_size = sizeof(struct ir_value *) * ra->num_regs;
  struct ir_value **active = alloca(active_size);

  if (active_in) {
    memcpy(active, active_in, active_size);
  } else {
    memset(active, 0, active_size);
  }

  list_for_each_entry_safe(instr, &block->instrs, struct ir_instr, it) {
    for (int i = 0; i < MAX_INSTR_ARGS; i++) {
      struct ir_value *arg = instr->arg[i];

      if (!arg || ir_is_constant(arg)) {
        continue;
      }

      /* make sure the argument is the current value in the register */
      CHECK_EQ(active[arg->reg], arg);
    }

    if (instr->result) {
      active[instr->result->reg] = instr->result;
    }
  }

  list_for_each_entry(edge, &block->outgoing, struct ir_edge, it) {
    ra_validate_r(ra, ir, edge->dst, active);
  }
}

static void ra_validate(struct ra *ra, struct ir *ir) {
  struct ir_block *head_block =
      list_first_entry(&ir->blocks, struct ir_block, it);
  ra_validate_r(ra, ir, head_block, NULL);
}

static void ra_pack_bin(struct ra *ra, struct bin *bin, struct tmp *tmp) {
  int reg = (int)(bin->reg - ra->regs);

  if (bin->tmp) {
    /* the existing temporary is no longer available in the bin's register */
    bin->tmp->value = NULL;
  }

  bin->tmp = tmp;

  if (bin->tmp) {
    /* assign the bin's register to the new temporary */
    bin->tmp->value->reg = reg;
  }
}

static int ra_alloc_blocked_reg(struct ra *ra, struct ir *ir, struct tmp *tmp) {
  /* find the register who's next use is furthest away */
  struct bin *spill_bin = NULL;
  int furthest_use = INT_MIN;

  for (int i = 0; i < ra->num_regs; i++) {
    struct bin *bin = &ra->bins[i];

    if (!bin->tmp) {
      continue;
    }

    if (!ra_reg_can_store(bin->reg, tmp->value)) {
      continue;
    }

    int next_use = bin->tmp->uses[bin->tmp->next];
    if (next_use > furthest_use) {
      furthest_use = next_use;
      spill_bin = bin;
    }
  }

  if (!spill_bin) {
    return 0;
  }

  /* spill the tmp if it wasn't previously spilled */
  struct tmp *spill_tmp = spill_bin->tmp;

  if (!spill_tmp->slot) {
    struct ir_instr *spill_after =
        list_prev_entry(tmp->value->def, struct ir_instr, it);
    struct ir_insert_point point = {tmp->value->def->block, spill_after};
    ir_set_insert_point(ir, &point);

    spill_tmp->slot = ir_alloc_local(ir, spill_tmp->value->type);
    ir_store_local(ir, spill_tmp->slot, spill_tmp->value);

    /* track spill stats */
    if (ir_is_int(spill_tmp->value->type)) {
      STAT_gprs_spilled++;
    } else {
      STAT_fprs_spilled++;
    }
  }

  /* assign temporary to spilled value's bin */
  ra_pack_bin(ra, spill_bin, tmp);

  return 1;
}

static int ra_alloc_free_reg(struct ra *ra, struct ir *ir, struct tmp *tmp) {
  /* find the first free register which can store the tmp's value */
  struct bin *alloc_bin = NULL;

  for (int i = 0; i < ra->num_regs; i++) {
    struct bin *bin = &ra->bins[i];

    if (bin->tmp) {
      continue;
    }

    if (!ra_reg_can_store(bin->reg, tmp->value)) {
      continue;
    }

    alloc_bin = bin;
    break;
  }

  if (!alloc_bin) {
    return 0;
  }

  /* assign the new tmp to the register's bin */
  ra_pack_bin(ra, alloc_bin, tmp);

  return 1;
}

static int ra_reuse_arg_reg(struct ra *ra, struct ir *ir, struct tmp *tmp) {
  struct ir_instr *instr = tmp->value->def;
  int pos = ra_ordinal(instr);

  if (!instr->arg[0] || ir_is_constant(instr->arg[0])) {
    return 0;
  }

  /* if the argument's register is used after this instruction, it's not
     trivial to reuse */
  struct tmp *arg = ra_tmp(instr->arg[0]);
  CHECK(arg->value && arg->value->reg != NO_REGISTER);

  if (arg->next != (arg->num_uses-1)) {
    return 0;
  }

  /* make sure the register can hold the tmp's value */
  struct bin *reuse_bin = &ra->bins[arg->value->reg];

  if (!ra_reg_can_store(reuse_bin->reg, tmp->value)) {
    return 0;
  }

  /* assign the new tmp to the register's bin */
  ra_pack_bin(ra, reuse_bin, tmp);

  return 1;
}

static void ra_alloc(struct ra *ra, struct ir *ir, struct ir_value *value) {
  if (!value) {
    return;
  }

  /* set initial value */
  struct tmp *tmp = ra_tmp(value);
  tmp->value = value;

  if (!ra_reuse_arg_reg(ra, ir, tmp)) {
    if (!ra_alloc_free_reg(ra, ir, tmp)) {
      if (!ra_alloc_blocked_reg(ra, ir, tmp)) {
        LOG_FATAL("Failed to allocate register");
      }
    }
  }
}

static void ra_rewrite_arg(struct ra *ra, struct ir *ir, struct ir_instr *instr,
                           int n) {
  struct ir_use *use = &instr->used[n];
  struct ir_value *value = *use->parg;

  if (!value || ir_is_constant(value)) {
    return;
  }

  struct tmp *tmp = ra_tmp(value);

  /* if the value isn't currently in a register, fill it from the stack */
  if (!tmp->value) {
    struct ir_instr *fill_after = list_prev_entry(instr, struct ir_instr, it);
    struct ir_insert_point point = {instr->block, fill_after};
    ir_set_insert_point(ir, &point);

    struct ir_value *fill = ir_load_local(ir, tmp->slot);
    int ordinal = ra_ordinal(instr);
    ra_set_ordinal(fill->def, ordinal - MAX_INSTR_ARGS + n);
    fill->tag = value->tag;
    tmp->value = fill;

    ra_alloc(ra, ir, fill);
  }

  /* replace original value with the tmp's latest value */
  CHECK_NOTNULL(tmp->value);
  ir_replace_use(use, tmp->value);
}

static void ra_expire_tmps(struct ra *ra, struct ir *ir,
                           struct ir_instr *current) {
  int current_ordinal = ra_ordinal(current);

  /* free up any bins which contain tmps that have now expired */
  for (int i = 0; i < ra->num_regs; i++) {
    struct bin *bin = &ra->bins[i];
    struct tmp *tmp = bin->tmp;

    if (!tmp) {
      continue;
    }

    while (1) {
      /* no more uses, expire temporary */
      if (tmp->next >= tmp->num_uses) {
        ra_pack_bin(ra, bin, NULL);
        break;
      }

      /* stop advancing once the next use is after the current position */
      int next_use = tmp->uses[tmp->next];
      if (next_use >= current_ordinal) {
        break;
      }

      tmp->next++;
    }
  }
}

static void ra_visit_r(struct ra *ra, struct ir *ir, struct ir_block *block) {
  /* work on a copy of the allocation state each recursion */
  size_t tmps_size = sizeof(struct tmp) * ra->num_tmps;
  size_t bins_size = sizeof(struct bin) * ra->num_regs;
  struct tmp *tmps_backup = alloca(tmps_size);
  struct bin *bins_backup = alloca(bins_size);
  memcpy(tmps_backup, ra->tmps, tmps_size);
  memcpy(bins_backup, ra->bins, bins_size);

  /* use safe iterator to avoid iterating over fills inserted
     when rewriting arguments */
  list_for_each_entry_safe(instr, &block->instrs, struct ir_instr, it) {
    ra_expire_tmps(ra, ir, instr);

    for (int i = 0; i < MAX_INSTR_ARGS; i++) {
      ra_rewrite_arg(ra, ir, instr, i);
    }

    ra_alloc(ra, ir, instr->result);
  }

  list_for_each_entry(edge, &block->outgoing, struct ir_edge, it) {
    ra_visit_r(ra, ir, edge->dst);
  }

  /* restore state */
  memcpy(ra->tmps, tmps_backup, tmps_size);
  memcpy(ra->bins, bins_backup, bins_size);
}

static void ra_visit(struct ra *ra, struct ir *ir) {
  struct ir_block *head_block =
      list_first_entry(&ir->blocks, struct ir_block, it);
  ra_visit_r(ra, ir, head_block);
}

static void ra_create_temporaries_r(struct ra *ra, struct ir *ir,
                                    struct ir_block *block) {
  list_for_each_entry(instr, &block->instrs, struct ir_instr, it) {
    int ordinal = ra_ordinal(instr);

    if (instr->result) {
      struct tmp *tmp = ra_create_tmp(ra, instr->result);
      ra_add_use(tmp, ordinal);
    }

    for (int i = 0; i < MAX_INSTR_ARGS; i++) {
      struct ir_value *arg = instr->arg[i];

      if (!arg || ir_is_constant(arg)) {
        continue;
      }

      struct tmp *tmp = ra_tmp(arg);
      ra_add_use(tmp, ordinal);
    }
  }

  list_for_each_entry(edge, &block->outgoing, struct ir_edge, it) {
    ra_create_temporaries_r(ra, ir, edge->dst);
  }
}

static void ra_create_temporaries(struct ra *ra, struct ir *ir) {
  struct ir_block *head_block =
      list_first_entry(&ir->blocks, struct ir_block, it);
  ra_create_temporaries_r(ra, ir, head_block);
}

static void ra_assign_ordinals_r(struct ra *ra, struct ir *ir,
                                 struct ir_block *block, int *ordinal) {
  /* assign each instruction an ordinal. these ordinals are used to describe
     the live range of a particular value */
  list_for_each_entry(instr, &block->instrs, struct ir_instr, it) {
    ra_set_ordinal(instr, *ordinal);

    /* each instruction could fill up to MAX_INSTR_ARGS, space out ordinals
       enough to allow for this */
    (*ordinal) += 1 + MAX_INSTR_ARGS;
  }

  list_for_each_entry(edge, &block->outgoing, struct ir_edge, it) {
    ra_assign_ordinals_r(ra, ir, edge->dst, ordinal);
  }
}

static void ra_assign_ordinals(struct ra *ra, struct ir *ir) {
  int ordinal = 0;
  struct ir_block *head_block =
      list_first_entry(&ir->blocks, struct ir_block, it);
  ra_assign_ordinals_r(ra, ir, head_block, &ordinal);
}

static void ra_reset(struct ra *ra, struct ir *ir) {
  for (int i = 0; i < ra->num_regs; i++) {
    struct bin *bin = &ra->bins[i];
    bin->tmp = NULL;
  }

  ra->num_tmps = 0;
}

void ra_run(struct ra *ra, struct ir *ir) {
  ra_reset(ra, ir);
  ra_assign_ordinals(ra, ir);
  ra_create_temporaries(ra, ir);
  ra_visit(ra, ir);

#if 1
  ra_validate(ra, ir);
#endif
}

void ra_destroy(struct ra *ra) {
  for (int i = 0; i < ra->max_tmps; i++) {
    struct tmp *tmp = &ra->tmps[i];
    free(tmp->uses);
  }
  free(ra->tmps);
  free(ra->bins);
  free(ra);
}

struct ra *ra_create(const struct jit_register *regs, int num_regs) {
  struct ra *ra = calloc(1, sizeof(struct ra));

  ra->regs = regs;
  ra->num_regs = num_regs;

  ra->bins = calloc(num_regs, sizeof(struct bin));

  for (int i = 0; i < ra->num_regs; i++) {
    struct bin *bin = &ra->bins[i];
    bin->reg = &ra->regs[i];
  }

  return ra;
}
