#ifndef JIT_H
#define JIT_H

#include <stdio.h>
#include "core/list.h"
#include "core/rb_tree.h"

struct address_space;
struct cfa;
struct cprop;
struct dce;
struct ir;
struct lse;
struct ra;
struct jit_code;

/* */
enum {
  BRANCH_FALL_THROUGH,
  BRANCH_STATIC,
  BRANCH_STATIC_TRUE,
  BRANCH_STATIC_FALSE,
  BRANCH_DYNAMIC,
  BRANCH_DYNAMIC_TRUE,
  BRANCH_DYNAMIC_FALSE,
};

struct jit_block_meta {
  /* address of block in guest memory */
  uint32_t guest_addr;

  /* destination address of terminating branch */
  int branch_type;
  uint32_t branch_addr;

  /* address of next instruction after branch */
  uint32_t next_addr;

  /* number of guest instructions in block */
  int num_instrs;

  /* estimated number of cycles to execute block */
  int num_cycles;

  /* size of block in bytes */
  int size;

  /* compilation units which use this meta data */
  struct list compile_refs;

  /* visit flag used when traversing block graph during compilation */
  unsigned visited;

  /* lookup map iterator for meta data cache */
  struct rb_node it;
};

/* intermediate structure to provide many to many relationship between
   block meta data and code, as well as misc. compile state */
struct jit_compile_unit {
  /* code that uses this compile unit */
  struct jit_code *parent;

  /* meta data to be compiled */
  struct jit_block_meta *meta;
  struct list_node meta_it;

  struct jit_compile_unit *branch;
  struct jit_compile_unit *next;
  struct ir_value *branch_cond;
  struct ir_value *branch_dest;
};

/* edges between compiled code, used to patch branches between them as they
   are compiled */
struct jit_edge {
  struct jit_code *src;
  struct jit_code *dst;

  /* location of branch instruction in host memory */
  void *branch;

  /* has this branch been patched */
  int patched;

  /* iterators for edge lists */
  struct list_node in_it;
  struct list_node out_it;
};

struct jit_code {
  /* address of entry point in guest memory */
  uint32_t guest_addr;

  /* use fastmem optimizations */
  int fastmem;

  /* guest code to be compiled */
  struct jit_compile_unit *root_unit;

  /* address of entry point in host memory */
  void *host_addr;
  int host_size;

  /* edges to other compiled code */
  struct list in_edges;
  struct list out_edges;

  /* lookup map iterators */
  struct rb_node it;
  struct rb_node rit;
};

struct jit_guest {
  /* memory interface */
  void *ctx;
  void *mem;
  struct address_space *space;
  uint8_t (*r8)(struct address_space *, uint32_t);
  uint16_t (*r16)(struct address_space *, uint32_t);
  uint32_t (*r32)(struct address_space *, uint32_t);
  uint64_t (*r64)(struct address_space *, uint32_t);
  void (*w8)(struct address_space *, uint32_t, uint8_t);
  void (*w16)(struct address_space *, uint32_t, uint16_t);
  void (*w32)(struct address_space *, uint32_t, uint32_t);
  void (*w64)(struct address_space *, uint32_t, uint64_t);

  /* dispatch interface */
  void *(*lookup_code)(uint32_t);
  void (*cache_code)(uint32_t, void *);
  void (*invalidate_code)(uint32_t);
  void (*patch_edge)(void *, void *);
  void (*restore_edge)(void *, uint32_t);
};

struct jit {
  char tag[32];

  struct jit_guest *guest;
  struct jit_frontend *frontend;
  struct jit_backend *backend;
  struct exception_handler *exc_handler;

  /* passes */
  struct cfa *cfa;
  struct lse *lse;
  struct cprop *cprop;
  struct esimp *esimp;
  struct dce *dce;
  struct ra *ra;

  /* scratch compilation buffer */
  uint8_t ir_buffer[1024 * 1024 * 16];

  /* block lookup trees */
  struct rb_tree meta;
  struct rb_tree code;
  struct rb_tree code_reverse;

  /* */
  unsigned visit_token;

  /* compiled block perf map */
  FILE *perf_map;

  /* dump ir to application directory as code compile */
  int dump_code;

  /* track emitter stats as code compile */
  int emit_stats;
};

struct jit *jit_create(const char *tag);
void jit_destroy(struct jit *jit);

int jit_init(struct jit *jit, struct jit_guest *guest,
             struct jit_frontend *frontend, struct jit_backend *backend);

void jit_compile_code(struct jit *jit, uint32_t guest_addr);
void jit_add_edge(struct jit *jit, void *code, uint32_t dst);

void jit_free_cache(struct jit *jit);
void jit_invalidate_cache(struct jit *jit);

#endif
