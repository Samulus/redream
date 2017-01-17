#ifndef ARMV3_FRONTEND_H
#define ARMV3_FRONTEND_H

#include "jit/frontend/jit_frontend.h"
#include "jit/jit.h"

struct armv3_frontend {
  struct jit_frontend;

  /* runtime interface */
  void *data;
  void (*translate)(void *, struct jit_code *, struct ir *);
  void (*switch_mode)(void *, uint32_t);
  void (*restore_mode)(void *);
  void (*software_interrupt)(void *);
};

struct armv3_frontend *armv3_frontend_create(struct jit *jit);
void armv3_frontend_destroy(struct armv3_frontend *frontend);

#endif
