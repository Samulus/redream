#include "hw/sh4/sh4.h"
#include "core/math.h"
#include "core/profiler.h"
#include "core/string.h"
#include "hw/aica/aica.h"
#include "hw/debugger.h"
#include "hw/dreamcast.h"
#include "hw/holly/holly.h"
#include "hw/memory.h"
#include "hw/pvr/pvr.h"
#include "hw/pvr/ta.h"
#include "hw/rom/boot.h"
#include "hw/rom/flash.h"
#include "hw/scheduler.h"
#include "jit/backend/x64/x64_backend.h"
#include "jit/frontend/sh4/sh4_frontend.h"
#include "jit/jit.h"
#include "sys/time.h"
#include "ui/nuklear.h"

DEFINE_PROF_STAT(sh4_instrs);
DEFINE_PROF_STAT(sh4_sr_updates);

/*
 * sh4 code layout. executable code sits between 0x0c000000 and 0x0d000000.
 * each instr is 2 bytes, making for a maximum of 0x1000000 >> 1 blocks
 */
#define SH4_BLOCK_MASK 0x03ffffff
#define SH4_BLOCK_SHIFT 1
#define SH4_BLOCK_OFFSET(addr) ((addr & SH4_BLOCK_MASK) >> SH4_BLOCK_SHIFT)
#define SH4_MAX_BLOCKS (0x1000000 >> SH4_BLOCK_SHIFT)

/* callbacks to service sh4_reg_read / sh4_reg_write calls */
struct reg_cb sh4_cb[NUM_SH4_REGS];

/*
 * global sh4 pointer is used by sh4_compile_pc to resolve the current
 * sh4 instance when compiling a new block
 */
static struct sh4 *g_sh4;

static void sh4_swap_gpr_bank(struct sh4 *sh4) {
  for (int s = 0; s < 8; s++) {
    uint32_t tmp = sh4->ctx.r[s];
    sh4->ctx.r[s] = sh4->ctx.ralt[s];
    sh4->ctx.ralt[s] = tmp;
  }
}

static void sh4_swap_fpr_bank(struct sh4 *sh4) {
  for (int s = 0; s <= 15; s++) {
    uint32_t tmp = sh4->ctx.fr[s];
    sh4->ctx.fr[s] = sh4->ctx.xf[s];
    sh4->ctx.xf[s] = tmp;
  }
}

static void sh4_invalid_instr(struct sh4_ctx *ctx, uint64_t data) {
  /*struct sh4 *self = reinterpret_cast<SH4 *>(ctx->sh4);
  uint32_t addr = (uint32_t)data;

  auto it = self->breakpoints.find(addr);
  CHECK_NE(it, self->breakpoints.end());

  // force the main loop to break
  self->ctx.num_cycles = 0;

  // let the debugger know execution has stopped
  self->dc->debugger->Trap();*/
}

void sh4_sr_updated(struct sh4_ctx *ctx, uint64_t old_sr) {
  struct sh4 *sh4 = ctx->sh4;

  STAT_sh4_sr_updates++;

  if ((ctx->sr & RB) != (old_sr & RB)) {
    sh4_swap_gpr_bank(sh4);
  }

  if ((ctx->sr & I) != (old_sr & I) || (ctx->sr & BL) != (old_sr & BL)) {
    sh4_intc_update_pending(sh4);
  }
}

void sh4_fpscr_updated(struct sh4_ctx *ctx, uint64_t old_fpscr) {
  struct sh4 *sh4 = ctx->sh4;

  if ((ctx->fpscr & FR) != (old_fpscr & FR)) {
    sh4_swap_fpr_bank(sh4);
  }
}

static uint32_t sh4_reg_read(struct sh4 *sh4, uint32_t addr, uint32_t data_mask) {
  uint32_t offset = SH4_REG_OFFSET(addr);
  reg_read_cb read = sh4_cb[offset].read;
  if (read) {
    return read(sh4->dc);
  }
  return sh4->reg[offset];
}

static void sh4_reg_write(struct sh4 *sh4, uint32_t addr, uint32_t data,
                   uint32_t data_mask) {
  uint32_t offset = SH4_REG_OFFSET(addr);
  reg_write_cb write = sh4_cb[offset].write;
  if (write) {
    write(sh4->dc, data);
    return;
  }
  sh4->reg[offset] = data;
}

void sh4_clear_interrupt(struct sh4 *sh4, enum sh4_interrupt intr) {
  sh4->requested_interrupts &= ~sh4->sort_id[intr];
  sh4_intc_update_pending(sh4);
}

void sh4_raise_interrupt(struct sh4 *sh4, enum sh4_interrupt intr) {
  sh4->requested_interrupts |= sh4->sort_id[intr];
  sh4_intc_update_pending(sh4);
}

void sh4_reset(struct sh4 *sh4, uint32_t pc) {
  /* unlink stale blocks */
  jit_unlink_blocks(sh4->jit);

  /* reset context */
  sh4->ctx.sh4 = sh4;
  sh4->ctx.InvalidInstruction = &sh4_invalid_instr;
  sh4->ctx.Prefetch = &sh4_ccn_prefetch;
  sh4->ctx.SRUpdated = &sh4_sr_updated;
  sh4->ctx.FPSCRUpdated = &sh4_fpscr_updated;
  sh4->ctx.pc = pc;
  sh4->ctx.r[15] = 0x8d000000;
  sh4->ctx.pr = 0x0;
  sh4->ctx.sr = 0x700000f0;
  sh4->ctx.fpscr = 0x00040001;

/* initialize registers */
#define SH4_REG(addr, name, default, type) \
  sh4->reg[name] = default;                \
  sh4->name = (type *)&sh4->reg[name];
#include "hw/sh4/sh4_regs.inc"
#undef SH4_REG

  /* reset interrupts */
  sh4_intc_reprioritize(sh4);

  sh4->execute_if->running = 1;
}

static void sh4_compile_pc() {
  uint32_t guest_addr = g_sh4->ctx.pc;
  uint8_t *guest_ptr = as_translate(g_sh4->memory_if->space, guest_addr);

  int flags = 0;
  if (g_sh4->ctx.fpscr & PR) {
    flags |= SH4_DOUBLE_PR;
  }
  if (g_sh4->ctx.fpscr & SZ) {
    flags |= SH4_DOUBLE_SZ;
  }

  code_pointer_t code = jit_compile_code(g_sh4->jit, guest_addr, flags);
  code();
}

static inline code_pointer_t sh4_get_code(struct sh4 *sh4, uint32_t addr) {
  int offset = SH4_BLOCK_OFFSET(addr);
  DCHECK_LT(offset, SH4_MAX_BLOCKS);
  return sh4->jit->code[offset];
}

static void sh4_run(struct device *dev, int64_t ns) {
  PROF_ENTER("cpu", "sh4_run");

  struct sh4 *sh4 = (struct sh4 *)dev;

  /* execute at least 1 cycle. the tests rely on this to step block by block */
  int64_t cycles = MAX(NANO_TO_CYCLES(ns, SH4_CLOCK_FREQ), INT64_C(1));

  /* each block's epilog will decrement the remaining cycles as they run */
  sh4->ctx.num_cycles = (int)cycles;

  g_sh4 = sh4;

  while (sh4->ctx.num_cycles > 0) {
    code_pointer_t code = sh4_get_code(sh4, sh4->ctx.pc);
    code();

    sh4_intc_check_pending(sh4);
  }

  g_sh4 = NULL;

  PROF_LEAVE();
}

static bool sh4_init(struct device *dev) {
  struct sh4 *sh4 = (struct sh4 *)dev;
  struct dreamcast *dc = sh4->dc;

  /* initialize jit interface */
  sh4->guest.block_mask = SH4_BLOCK_MASK;
  sh4->guest.block_shift = SH4_BLOCK_SHIFT;
  sh4->guest.block_max = SH4_MAX_BLOCKS;
  sh4->guest.ctx_base = &sh4->ctx;
  sh4->guest.mem_base = sh4->memory_if->space->base;
  sh4->guest.mem_self = sh4->memory_if->space;
  sh4->guest.r8 = &as_read8;
  sh4->guest.r16 = &as_read16;
  sh4->guest.r32 = &as_read32;
  sh4->guest.w8 = &as_write8;
  sh4->guest.w16 = &as_write16;
  sh4->guest.w32 = &as_write32;

  sh4->jit_frontend = sh4_frontend_create(&sh4->guest);
  sh4->jit_backend = x64_backend_create(&sh4->guest);
  sh4->jit = jit_create(&sh4->guest, sh4->jit_frontend, sh4->jit_backend,
                        &sh4_compile_pc, "sh4");

  return true;
}

void sh4_destroy(struct sh4 *sh4) {
  if (sh4->jit) {
    jit_destroy(sh4->jit);
  }

  if (sh4->jit_backend) {
    x64_backend_destroy(sh4->jit_backend);
  }

  if (sh4->jit_backend) {
    sh4_frontend_destroy(sh4->jit_frontend);
  }

  dc_destroy_memory_interface(sh4->memory_if);
  dc_destroy_execute_interface(sh4->execute_if);
  dc_destroy_device((struct device *)sh4);
}

struct sh4 *sh4_create(struct dreamcast *dc) {
  struct sh4 *sh4 = dc_create_device(dc, sizeof(struct sh4), "sh", &sh4_init);
  sh4->execute_if = dc_create_execute_interface(&sh4_run, 0);
  sh4->memory_if = dc_create_memory_interface(dc, &sh4_data_map);

  return sh4;
}

REG_R32(sh4_cb, PDTRA) {
  struct sh4 *sh4 = dc->sh4;
  /*
   * magic values to get past 0x8c00b948 in the boot rom:
   * void _8c00b92c(int arg1) {
   *   sysvars->var1 = reg[PDTRA];
   *   for (i = 0; i < 4; i++) {
   *     sysvars->var2 = reg[PDTRA];
   *     if (arg1 == sysvars->var2 & 0x03) {
   *       return;
   *     }
   *   }
   *   reg[PR] = (uint32_t *)0x8c000000;
   * }
   * old_PCTRA = reg[PCTRA];
   * i = old_PCTRA | 0x08;
   * reg[PCTRA] = i;
   * reg[PDTRA] = reg[PDTRA] | 0x03;
   * _8c00b92c(3);
   * reg[PCTRA] = i | 0x03;
   * _8c00b92c(3);
   * reg[PDTRA] = reg[PDTRA] & 0xfffe;
   * _8c00b92c(0);
   * reg[PCTRA] = i;
   * _8c00b92c(3);
   * reg[PCTRA] = i | 0x04;
   * _8c00b92c(3);
   * reg[PDTRA] = reg[PDTRA] & 0xfffd;
   * _8c00b92c(0);
   * reg[PCTRA] = old_PCTRA;
   */
  uint32_t pctra = *sh4->PCTRA;
  uint32_t pdtra = *sh4->PDTRA;
  uint32_t v = 0;
  if ((pctra & 0xf) == 0x8 || ((pctra & 0xf) == 0xb && (pdtra & 0xf) != 0x2) ||
      ((pctra & 0xf) == 0xc && (pdtra & 0xf) == 0x2)) {
    v = 3;
  }
  /*
   * FIXME cable setting
   * When a VGA cable* is connected
   * 1. The SH4 obtains the cable information from the PIO port.  (PB[9:8] =
   * "00")
   * 2. Set the HOLLY synchronization register for VGA.  (The SYNC output is
   * H-Sync and V-Sync.)
   * 3. When VREG1 = 0 and VREG0 = 0 are written in the AICA register,
   * VIDEO1 = 0 and VIDEO0 = 1 are output.  VIDEO0 is connected to the
   * DVE-DACH pin, and handles switching between RGB and NTSC/PAL.
   *
   * When an RGB(NTSC/PAL) cable* is connected
   * 1. The SH4 obtains the cable information from the PIO port.  (PB[9:8] =
   * "10")
   * 2. Set the HOLLY synchronization register for NTSC/PAL.  (The SYNC
   * output is H-Sync and V-Sync.)
   * 3. When VREG1 = 0 and VREG0 = 0 are written in the AICA register,
   * VIDEO1 = 1 and VIDEO0 = 0 are output.  VIDEO0 is connected to the
   * DVE-DACH pin, and handles switching between RGB and NTSC/PAL.
   *
   * When a stereo A/V cable, an S-jack cable* or an RF converter* is
   * connected
   * 1. The SH4 obtains the cable information from the PIO port.  (PB[9:8] =
   * "11")
   * 2. Set the HOLLY synchronization register for NTSC/PAL.  (The SYNC
   * output is H-Sync and V-Sync.)
   * 3. When VREG1 = 1 and VREG0 = 1 are written in the AICA register,
   * VIDEO1 = 0 and VIDEO0 = 0 are output.  VIDEO0 is connected to the
   * DVE-DACH pin, and handles switching between RGB and NTSC/PAL.
   */
  v |= 0x3 << 8;
  return v;
}

// clang-format off
AM_BEGIN(struct sh4, sh4_data_map)
  AM_RANGE(0x00000000, 0x001fffff) AM_DEVICE("boot", boot_rom_map)
  AM_RANGE(0x00200000, 0x0021ffff) AM_DEVICE("flash", flash_rom_map)
  AM_RANGE(0x0c000000, 0x0cffffff) AM_MOUNT("system ram")

  /* main ram mirrors */
  AM_RANGE(0x0d000000, 0x0dffffff) AM_MIRROR(0x0c000000)
  AM_RANGE(0x0e000000, 0x0effffff) AM_MIRROR(0x0c000000)
  AM_RANGE(0x0f000000, 0x0fffffff) AM_MIRROR(0x0c000000)

  /* external devices */
  AM_RANGE(0x005f6000, 0x005f7fff) AM_DEVICE("holly", holly_reg_map)
  AM_RANGE(0x005f8000, 0x005f9fff) AM_DEVICE("pvr", pvr_reg_map)
  AM_RANGE(0x00600000, 0x0067ffff) AM_DEVICE("holly", holly_modem_map)
  AM_RANGE(0x00700000, 0x00710fff) AM_DEVICE("aica", aica_reg_map)
  AM_RANGE(0x00800000, 0x00ffffff) AM_DEVICE("aica", aica_data_map)
  AM_RANGE(0x01000000, 0x01ffffff) AM_DEVICE("holly", holly_expansion0_map)
  AM_RANGE(0x02700000, 0x02ffffff) AM_DEVICE("holly", holly_expansion1_map)
  AM_RANGE(0x04000000, 0x057fffff) AM_DEVICE("pvr", pvr_vram_map)
  AM_RANGE(0x10000000, 0x11ffffff) AM_DEVICE("ta", ta_fifo_map)
  AM_RANGE(0x14000000, 0x17ffffff) AM_DEVICE("holly", holly_expansion2_map)

  /* internal registers */
  AM_RANGE(0x1e000000, 0x1fffffff) AM_HANDLE("sh4 reg",
                                             (mmio_read_cb)&sh4_reg_read,
                                             (mmio_write_cb)&sh4_reg_write)

  /* physical mirrors */
  AM_RANGE(0x20000000, 0x3fffffff) AM_MIRROR(0x00000000)  /* p0 */
  AM_RANGE(0x40000000, 0x5fffffff) AM_MIRROR(0x00000000)  /* p0 */
  AM_RANGE(0x60000000, 0x7fffffff) AM_MIRROR(0x00000000)  /* p0 */
  AM_RANGE(0x80000000, 0x9fffffff) AM_MIRROR(0x00000000)  /* p1 */
  AM_RANGE(0xa0000000, 0xbfffffff) AM_MIRROR(0x00000000)  /* p2 */
  AM_RANGE(0xc0000000, 0xdfffffff) AM_MIRROR(0x00000000)  /* p3 */
  AM_RANGE(0xe0000000, 0xffffffff) AM_MIRROR(0x00000000)  /* p4 */

  /* internal cache and sq only accessible through p4 */
  AM_RANGE(0x7c000000, 0x7fffffff) AM_HANDLE("sh4 cache",
                                             (mmio_read_cb)&sh4_ccn_cache_read,
                                             (mmio_write_cb)&sh4_ccn_cache_write)
  AM_RANGE(0xe0000000, 0xe3ffffff) AM_HANDLE("sh4 sq",
                                             (mmio_read_cb)&sh4_ccn_sq_read,
                                             (mmio_write_cb)&sh4_ccn_sq_write)
AM_END();
// clang-format on

