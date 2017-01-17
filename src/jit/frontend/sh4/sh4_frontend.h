#ifndef SH4_FRONTEND_H
#define SH4_FRONTEND_H

#include "jit/frontend/jit_frontend.h"

struct ir;
struct jit;
struct jit_block_meta;

enum {
  SH4_FASTMEM = 0x1,
  SH4_DOUBLE_PR = 0x2,
  SH4_DOUBLE_SZ = 0x4,
  SH4_SINGLE_INSTR = 0x8,
};

struct sh4_frontend {
  struct jit_frontend;

  /* runtime interface */
  void *data;
  void (*translate)(void *, struct jit_code *, struct ir *);
  void (*invalid_instr)(void *, uint32_t);
  void (*sq_prefetch)(void *, uint32_t);
  void (*sr_updated)(void *, uint32_t);
  void (*fpscr_updated)(void *, uint32_t);
};

struct sh4_frontend *sh4_frontend_create(struct jit *jit);
void sh4_frontend_destroy(struct sh4_frontend *frontend);

#endif
