#ifndef SH4_TRANSLATE_H
#define SH4_TRANSLATE_H

struct ir;
struct jit_compile_unit;
struct sh4_instr;
struct sh4_frontend;

void sh4_emit_instr(struct sh4_frontend *frontend, struct jit_compile_unit *unit, struct ir *ir, int flags,
                    const struct sh4_instr *instr,
                    const struct sh4_instr *delay);

#endif
