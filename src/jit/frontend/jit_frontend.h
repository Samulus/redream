#ifndef JIT_FRONTEND_H
#define JIT_FRONTEND_H

#include <stdint.h>

struct ir;
struct jit;
struct jit_code;
struct jit_frontend;
struct jit_block_meta;

struct jit_frontend {
  struct jit *jit;
  int (*analyze_code)(struct jit_frontend *base, struct jit_block_meta *meta);
  void (*translate_code)(struct jit_frontend *base, struct jit_code *code, struct ir *ir);
  void (*dump_code)(struct jit_frontend *base, uint32_t addr, int size);
};

#endif
