#ifndef HOLLY_H
#define HOLLY_H

#include <stdint.h>
#include "hw/dreamcast.h"
#include "hw/holly/holly_types.h"
#include "hw/memory.h"

struct dreamcast;
struct gdrom;
struct maple;
struct sh4;

struct holly {
  struct device;
  uint32_t reg[NUM_HOLLY_REGS];

#define HOLLY_REG(offset, name, default, type) type *name;
#include "hw/holly/holly_regs.inc"
#undef HOLLY_REG
};

AM_DECLARE(holly_reg_map);
AM_DECLARE(holly_modem_map);
AM_DECLARE(holly_expansion0_map);
AM_DECLARE(holly_expansion1_map);
AM_DECLARE(holly_expansion2_map);

extern struct reg_cb holly_cb[NUM_HOLLY_REGS];

struct holly *holly_create(struct dreamcast *dc);
void holly_destroy(struct holly *hl);

void holly_raise_interrupt(struct holly *hl, holly_interrupt_t intr);
void holly_clear_interrupt(struct holly *hl, holly_interrupt_t intr);

#endif
