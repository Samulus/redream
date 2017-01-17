#include <inttypes.h>
#include "jit/jit.h"
#include "core/core.h"
#include "core/option.h"
#include "core/profiler.h"
#include "jit/backend/jit_backend.h"
#include "jit/frontend/jit_frontend.h"
#include "jit/ir/ir.h"
#include "jit/passes/constant_propagation_pass.h"
#include "jit/passes/control_flow_analysis_pass.h"
#include "jit/passes/dead_code_elimination_pass.h"
#include "jit/passes/expression_simplification_pass.h"
#include "jit/passes/load_store_elimination_pass.h"
#include "jit/passes/register_allocation_pass.h"
#include "sys/exception_handler.h"
#include "sys/filesystem.h"

#if PLATFORM_DARWIN || PLATFORM_LINUX
#include <unistd.h>
#endif

DEFINE_OPTION_INT(perf, 0, "Generate perf-compatible maps for genrated code");

static int meta_map_cmp(const struct rb_node *rb_lhs,
                         const struct rb_node *rb_rhs) {
  const struct jit_block_meta *lhs =
      container_of(rb_lhs, const struct jit_block_meta, it);
  const struct jit_block_meta *rhs =
      container_of(rb_rhs, const struct jit_block_meta, it);

  if (lhs->guest_addr < rhs->guest_addr) {
    return -1;
  } else if (lhs->guest_addr > rhs->guest_addr) {
    return 1;
  } else {
    return 0;
  }
}

static int code_map_cmp(const struct rb_node *rb_lhs,
                         const struct rb_node *rb_rhs) {
  const struct jit_code *lhs =
      container_of(rb_lhs, const struct jit_code, it);
  const struct jit_code *rhs =
      container_of(rb_rhs, const struct jit_code, it);

  if (lhs->guest_addr < rhs->guest_addr) {
    return -1;
  } else if (lhs->guest_addr > rhs->guest_addr) {
    return 1;
  } else {
    return 0;
  }
}

static int code_reverse_map_cmp(const struct rb_node *rb_lhs,
                                 const struct rb_node *rb_rhs) {
  const struct jit_code *lhs =
      container_of(rb_lhs, const struct jit_code, rit);
  const struct jit_code *rhs =
      container_of(rb_rhs, const struct jit_code, rit);

  if ((uint8_t *)lhs->host_addr < (uint8_t *)rhs->host_addr) {
    return -1;
  } else if ((uint8_t *)lhs->host_addr > (uint8_t *)rhs->host_addr) {
    return 1;
  } else {
    return 0;
  }
}

static struct rb_callbacks meta_map_cb = {
    &meta_map_cmp, NULL, NULL,
};

static struct rb_callbacks code_map_cb = {
    &code_map_cmp, NULL, NULL,
};

static struct rb_callbacks code_reverse_map_cb = {
    &code_reverse_map_cmp, NULL, NULL,
};

static struct jit_block_meta *jit_lookup_meta(struct jit *jit, uint32_t guest_addr) {
  struct jit_block_meta search;
  search.guest_addr = guest_addr;

  return rb_find_entry(&jit->meta, &search, struct jit_block_meta, it,
                       &meta_map_cb);
}

static struct jit_code *jit_lookup_code(struct jit *jit, uint32_t guest_addr) {
  struct jit_code search;
  search.guest_addr = guest_addr;

  return rb_find_entry(&jit->code, &search, struct jit_code, it,
                       &code_map_cb);
}

static struct jit_code *jit_lookup_code_reverse(struct jit *jit,
                                                  void *host_addr) {
  /* when performing a reverse lookup, host_addr represents an address
     somewhere within a block, not necessarily the start of the block */
  struct jit_code search;
  search.host_addr = host_addr;

  struct rb_node *first = rb_first(&jit->code_reverse);
  struct rb_node *last = rb_last(&jit->code_reverse);
  struct rb_node *rit =
      rb_upper_bound(&jit->code_reverse, &search.rit, &code_reverse_map_cb);

  if (rit == first) {
    return NULL;
  }

  rit = rit ? rb_prev(rit) : last;

  struct jit_code *code = container_of(rit, struct jit_code, rit);
  if ((uint8_t *)host_addr < (uint8_t *)code->host_addr ||
      (uint8_t *)host_addr >=
          ((uint8_t *)code->host_addr + code->host_size)) {
    return NULL;
  }

  return code;
}

static int jit_is_stale(struct jit *jit, struct jit_code *code) {
  void *ptr = jit->guest->lookup_code(code->guest_addr);
  return ptr != code->host_addr;
}

static void jit_patch_edges(struct jit *jit, struct jit_code *code) {
  PROF_ENTER("cpu", "jit_patch_edges");

  /* patch incoming edges to this block to directly jump to it instead of
     going through dispatch */
  list_for_each_entry(edge, &code->in_edges, struct jit_edge, in_it) {
    if (!edge->patched) {
      edge->patched = 1;
      jit->guest->patch_edge(edge->branch, edge->dst->host_addr);
    }
  }

  /* patch outgoing edges to other code at this time */
  list_for_each_entry(edge, &code->out_edges, struct jit_edge, out_it) {
    if (!edge->patched) {
      edge->patched = 1;
      jit->guest->patch_edge(edge->branch, edge->dst->host_addr);
    }
  }

  PROF_LEAVE();
}

static void jit_restore_edges(struct jit *jit, struct jit_code *code) {
  PROF_ENTER("cpu", "jit_restore_edges");

  /* restore any patched branches to go back through dispatch */
  list_for_each_entry(edge, &code->in_edges, struct jit_edge, in_it) {
    if (edge->patched) {
      edge->patched = 0;
      jit->guest->restore_edge(edge->branch, edge->dst->guest_addr);
    }
  }

  PROF_LEAVE();
}

static void jit_finalize_code(struct jit *jit, struct jit_code *code) {
  CHECK(list_empty(&code->in_edges) && list_empty(&code->out_edges), "code shouldn't have any existing edges");
  CHECK(rb_empty_node(&code->it) && rb_empty_node(&code->rit), "code was already inserted in ldokup tables");

  jit->guest->cache_code(code->guest_addr, code->host_addr);

  rb_insert(&jit->code, &code->it, &code_map_cb);
  rb_insert(&jit->code_reverse, &code->rit, &code_reverse_map_cb);

  /* write out to perf map if enabled */
  if (OPTION_perf) {
    fprintf(jit->perf_map, "%" PRIxPTR " %x %s_0x%08x\n",
            (uintptr_t)code->host_addr, code->host_size, jit->tag, code->guest_addr);
  }
}

static void jit_free_compile_unit(struct jit *jit, struct jit_compile_unit **unit_ptr) {
  struct jit_compile_unit *unit = *unit_ptr;

  if (!unit) {
    return;
  }

  jit_free_compile_unit(jit, &unit->branch);
  jit_free_compile_unit(jit, &unit->next);

  list_remove(&unit->meta->compile_refs, &unit->meta_it);
  free(unit);
  *unit_ptr = NULL;
}

static void jit_invalidate_code(struct jit *jit, struct jit_code *code) {
  /* FIXME */
  jit_free_compile_unit(jit, &code->root_unit);

  /* invalid code from guest dispatch cache and remove any direct branches
     to this code */
  jit->guest->invalidate_code(code->guest_addr);

  jit_restore_edges(jit, code);

  list_for_each_entry_safe(edge, &code->in_edges, struct jit_edge, in_it) {
    list_remove(&edge->src->out_edges, &edge->out_it);
    list_remove(&code->in_edges, &edge->in_it);
    free(edge);
  }

  list_for_each_entry_safe(edge, &code->out_edges, struct jit_edge, out_it) {
    list_remove(&code->out_edges, &edge->out_it);
    list_remove(&edge->dst->in_edges, &edge->in_it);
    free(edge);
  }
}

static void jit_free_code(struct jit *jit, struct jit_code *code) {
  jit_invalidate_code(jit, code);

  if (!rb_empty_node(&code->it)) {
    rb_unlink(&jit->code, &code->it, &code_map_cb);
  }
  if (!rb_empty_node(&code->rit)) {
    rb_unlink(&jit->code_reverse, &code->rit, &code_reverse_map_cb);
  }

  free(code);
}

static struct jit_code *jit_alloc_code(struct jit *jit) {
  struct jit_code *code = calloc(1, sizeof(struct jit_code));

  return code;
}

static void jit_free_meta(struct jit *jit, struct jit_block_meta *meta) {
  /* FIXME */
  CHECK(list_empty(&meta->compile_refs), "code most be freed before meta data");

  rb_unlink(&jit->meta, &meta->it, &meta_map_cb);

  free(meta);
}

static struct jit_block_meta *jit_alloc_meta(struct jit *jit, uint32_t addr) {
  struct jit_block_meta *meta = calloc(1, sizeof(struct jit_block_meta));
  meta->guest_addr = addr;
  meta->branch_addr = INVALID_ADDR;
  meta->next_addr = INVALID_ADDR;

  rb_insert(&jit->meta, &meta->it, &meta_map_cb);

  return meta;
}

/* invalidate code pointers, but don't remove code entries from lookup maps.
   this is used when clearing the jit while code is currently executing */
void jit_invalidate_cache(struct jit *jit) {
  /* FIXME */
  rb_for_each_entry_safe(code, &jit->code, struct jit_code, it) {
    jit_invalidate_code(jit, code);
  }

  rb_for_each_entry_safe(meta, &jit->meta, struct jit_block_meta, it) {
    jit_free_meta(jit, meta);
  }
  CHECK(rb_empty_tree(&jit->meta));
}

 /* invalidate code pointers and remove code entries from lookup maps. this
    is only safe to use when no code is currently executing */
void jit_free_cache(struct jit *jit) {
  /* FIXME */
  jit_invalidate_cache(jit);

  rb_for_each_entry_safe(code, &jit->code, struct jit_code, it) {
    jit_free_code(jit, code);
  }
  CHECK(rb_empty_tree(&jit->code));
  CHECK(rb_empty_tree(&jit->code_reverse));
}

void jit_add_edge(struct jit *jit, void *branch, uint32_t addr) {
  struct jit_code *src = jit_lookup_code_reverse(jit, branch);
  struct jit_code *dst = jit_lookup_code(jit, addr);

  if (jit_is_stale(jit, src) || !dst) {
    return;
  }

  struct jit_edge *edge = calloc(1, sizeof(struct jit_edge));
  edge->src = src;
  edge->dst = dst;
  edge->branch = branch;
  list_add(&src->out_edges, &edge->out_it);
  list_add(&dst->in_edges, &edge->in_it);

  jit_patch_edges(jit, src);
}

static void jit_dump_code(struct jit *jit, uint32_t guest_addr,
                           struct ir *ir) {
  const char *appdir = fs_appdir();

  char irdir[PATH_MAX];
  snprintf(irdir, sizeof(irdir), "%s" PATH_SEPARATOR "ir", appdir);
  CHECK(fs_mkdir(irdir));

  char filename[PATH_MAX];
  snprintf(filename, sizeof(filename), "%s" PATH_SEPARATOR "0x%08x.ir", irdir,
           guest_addr);

  FILE *file = fopen(filename, "w");
  CHECK_NOTNULL(file);
  ir_write(ir, file);
  fclose(file);
}

static struct jit_compile_unit *jit_analyze_code_r(struct jit *jit, struct jit_code *code, uint32_t guest_addr) {
  struct jit_block_meta *meta = jit_lookup_meta(jit, guest_addr);

  /* don't allow control flow to rejoin */
  if (meta && meta->visited == jit->visit_token) {
    return NULL;
  }

  /* FIXME cleanup */
  if (!meta) {
    meta = jit_alloc_meta(jit, guest_addr);

    /* analyze fails currently during bootstrap when a branch is found, but the
       code actually hasn't been written out to memory just yet */
    if (!jit->frontend->analyze_code(jit->frontend, meta)) {
      jit_free_meta(jit, meta);
      return NULL;
    }
  }

  meta->visited = jit->visit_token;

  struct jit_compile_unit *unit = calloc(1, sizeof(struct jit_compile_unit));
  unit->parent = code;
  unit->meta = meta;
  list_add(&meta->compile_refs, &unit->meta_it);

  if (meta->branch_addr != INVALID_ADDR) {
    unit->branch = jit_analyze_code_r(jit, code, meta->branch_addr);
  }

  if (meta->next_addr != INVALID_ADDR) {
    unit->next = jit_analyze_code_r(jit, code, meta->next_addr);
  }

  return unit;
}

static void jit_analyze_code(struct jit *jit, struct jit_code *code) {
  jit->visit_token++;
  code->root_unit = jit_analyze_code_r(jit, code, code->guest_addr);
  CHECK_NOTNULL(code->root_unit);
}

void jit_compile_code(struct jit *jit, uint32_t guest_addr) {
  PROF_ENTER("cpu", "jit_compile_code");

#if 1
  LOG_INFO("jit_compile_code %s 0x%08x", jit->tag, guest_addr);
#endif

  /* for debug builds, fastmem can be troublesome when running under gdb or
     lldb. when doing so, SIGSEGV handling can be completely disabled with:
     handle SIGSEGV nostop noprint pass
     however, then legitimate SIGSEGV will also not be handled by the debugger.
     as of this writing, there is no way to configure the debugger to ignore the
     signal initially, letting us try to handle it, and then handling it in the
     case that we do not (e.g. because it was not a fastmem-related segfault).
     because of this, fastmem is default disabled for debug builds to cause less
     headaches */
  int fastmem = 1;
#ifndef NDEBUG
  fastmem = 0;
#endif

  /* if the address being compiled had previously been invalidated by a fastmem
     exception, finish freeing it at this time and disable fastmem opts for the
     new block */
  struct jit_code *existing = jit_lookup_code(jit, guest_addr);

  if (existing) {
    fastmem = existing->fastmem;
    jit_free_code(jit, existing);
  }

  /* create the new code entry */
  struct jit_code *code = jit_alloc_code(jit);
  code->guest_addr = guest_addr;
  code->fastmem = fastmem;

  /* analyze the guest address, adding translion units to the new code entry */
  jit_analyze_code(jit, code);

  /* translate the source machine code into ir */
  struct ir ir = {0};
  ir.buffer = jit->ir_buffer;
  ir.capacity = sizeof(jit->ir_buffer);

  jit->frontend->translate_code(jit->frontend, code, &ir);

  /* dump unoptimized block */
  if (jit->dump_code) {
    jit_dump_code(jit, guest_addr, &ir);
  }

  //ir_write(&ir, stdout);

  /* run optimization passes */
  cfa_run(jit->cfa, &ir);
  lse_run(jit->lse, &ir);
  cprop_run(jit->cprop, &ir);
  esimp_run(jit->esimp, &ir);
  dce_run(jit->dce, &ir);
  ra_run(jit->ra, &ir);

  /* assemble the ir into native code */
  int res = jit->backend->assemble_code(jit->backend, code, &ir);

  if (res) {
    jit_finalize_code(jit, code);
  } else {
    /* if the backend overflowed, completely free the cache and let dispatch
       try to compile again */
    LOG_INFO("backend overflow, resetting code cache");
    jit_free_cache(jit);
  }

  PROF_LEAVE();
}

static int jit_handle_exception(void *data, struct exception *ex) {
  struct jit *jit = data;

  /* see if there is a cached block corresponding to the current pc */
  struct jit_code *code = jit_lookup_code_reverse(jit, (void *)ex->pc);

  if (!code) {
    return 0;
  }

  /* let the backend attempt to handle the exception */
  if (!jit->backend->handle_exception(jit->backend, ex)) {
    return 0;
  }

  /* invalidate the block so it's recompiled without fastmem optimizations
     on the next access. note, the block can't be removed from the lookup
     maps at this point because it's still executing and may raise more
     exceptions */
  code->fastmem = 0;
  jit_invalidate_code(jit, code);

  return 1;
}

int jit_init(struct jit *jit, struct jit_guest *guest,
             struct jit_frontend *frontend, struct jit_backend *backend) {
  jit->guest = guest;
  jit->frontend = frontend;
  jit->backend = backend;
  jit->exc_handler = exception_handler_add(jit, &jit_handle_exception);

  jit->lse = lse_create();
  jit->cprop = cprop_create();
  jit->esimp = esimp_create();
  jit->dce = dce_create();
  jit->ra = ra_create(jit->backend->registers, jit->backend->num_registers);

  /* open perf map if enabled */
  if (OPTION_perf) {
#if PLATFORM_DARWIN || PLATFORM_LINUX
    char perf_map_path[PATH_MAX];
    snprintf(perf_map_path, sizeof(perf_map_path), "/tmp/perf-%d.map",
             getpid());
    jit->perf_map = fopen(perf_map_path, "a");
    CHECK_NOTNULL(jit->perf_map);
#endif
  }

  return 1;
}

void jit_destroy(struct jit *jit) {
  if (OPTION_perf) {
    if (jit->perf_map) {
      fclose(jit->perf_map);
    }
  }

  if (jit->ra) {
    ra_destroy(jit->ra);
  }

  if (jit->dce) {
    dce_destroy(jit->dce);
  }

  if (jit->esimp) {
    esimp_destroy(jit->esimp);
  }

  if (jit->cprop) {
    cprop_destroy(jit->cprop);
  }

  if (jit->lse) {
    lse_destroy(jit->lse);
  }

  if (jit->exc_handler) {
    exception_handler_remove(jit->exc_handler);
  }

  if (jit->backend) {
    jit_free_cache(jit);
  }

  free(jit);
}

struct jit *jit_create(const char *tag) {
  struct jit *jit = calloc(1, sizeof(struct jit));

  strncpy(jit->tag, tag, sizeof(jit->tag));

  return jit;
}
