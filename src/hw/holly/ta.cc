#include <imgui.h>
#include <string.h>
#include "core/list.h"
#include "core/profiler.h"
#include "core/rb_tree.h"
#include "hw/holly/holly.h"
#include "hw/holly/pixel_convert.h"
#include "hw/holly/pvr.h"
#include "hw/holly/ta.h"
#include "hw/holly/tr.h"
// #include "hw/holly/trace.h"
#include "hw/sh4/sh4.h"
#include "renderer/backend.h"
#include "sys/exception_handler.h"

static const int MAX_TEXTURE_ENTRIES = 1024;
static const int MAX_TILE_CONTEXTS = 4;

typedef struct {
  struct ta_s *ta;
  texture_key_t key;
  texture_handle_t handle;
  struct memory_watch_s *texture_watch;
  struct memory_watch_s *palette_watch;
  list_node_t free_it;
  rb_node_t live_it;
  list_node_t invalid_it;
} texture_entry_t;

typedef struct ta_s {
  device_t base;

  struct holly_s *holly;
  struct pvr_s *pvr;
  struct address_space_s *space;
  struct rb_s *rb;
  struct tr_s *tr;
  uint8_t *video_ram;

  // texture cache entry pool. free entries are in a linked list, live entries
  // are in a tree ordered by texture key, textures queued for invalidation are
  // in the the invalid_entries linked list
  texture_entry_t entries[MAX_TEXTURE_ENTRIES];
  list_t free_entries;
  rb_tree_t live_entries;
  list_t invalid_entries;
  int num_invalidated;

  // tile context pool. free contexts are in a linked list, live contexts are
  // are in a tree ordered by the context's guest address, and a pointer to the
  // next context up for rendering is stored in pending_context
  ta_ctx_t contexts[MAX_TILE_CONTEXTS];
  list_t free_contexts;
  rb_tree_t live_contexts;
  ta_ctx_t *pending_context;

  // persistent tile render context. parsing the tile context involves growing
  // dynamic arrays for surfaces and vertices, keeping a persistent context
  // keeps us from allocating and growing these each frame
  tr_ctx_t rctx;

  // TraceWriter *trace_writer;
} ta_t;

static void ta_build_tables();
static int ta_get_param_size_raw(pcw_t pcw, int vertex_type);
static int ta_get_poly_type_raw(pcw_t pcw);
static int ta_get_vert_type_raw(pcw_t pcw);

static int ta_entry_cmp(const rb_node_t *lhs_it, const rb_node_t *rhs_it);
static int ta_context_cmp(const rb_node_t *lhs_it, const rb_node_t *rhs_it);

static bool ta_init(ta_t *ta);
static void ta_paint(ta_t *ta, bool show_main_menu);
static texture_entry_t *ta_alloc_entry(ta_t *ta, texture_key_t key);
static void ta_free_entry(ta_t *ta, texture_entry_t *entry);
static texture_handle_t ta_get_texture(ta_t *ta, const ta_ctx_t *ctx, tsp_t tsp,
                                       tcw_t tcw, void *register_data,
                                       register_texture_cb register_cb);
static void ta_clear_textures(ta_t *ta);
static void ta_clear_pending_textures(ta_t *ta);
static void ta_invalidate_texture(ta_t *ta, texture_entry_t *entry);
static void ta_write_texture(const re_exception_t *ex, texture_entry_t *entry);
static void ta_write_palette(const re_exception_t *ex, texture_entry_t *entry);
static void ta_soft_reset(ta_t *ta);
static ta_ctx_t *ta_get_context(ta_t *ta, uint32_t addr);
static ta_ctx_t *ta_alloc_context(ta_t *ta, uint32_t addr);
static void ta_unlink_context(ta_t *ta, ta_ctx_t *ctx);
static void ta_free_context(ta_t *ta, ta_ctx_t *ctx);
static void ta_init_context(ta_t *ta, uint32_t addr);
static void ta_write_context(ta_t *ta, uint32_t addr, uint32_t value);
static void ta_finish_context(ta_t *ta, uint32_t addr);
static void ta_save_state(ta_t *ta, ta_ctx_t *tctx);
static void ta_write_poly_fifo(ta_t *ta, uint32_t addr, uint32_t value);
static void ta_write_texture_fifo(ta_t *ta, uint32_t addr, uint32_t value);
DECLARE_REG_W32(ta_t *ta, SOFTRESET);
DECLARE_REG_W32(ta_t *ta, TA_LIST_INIT);
DECLARE_REG_W32(ta_t *ta, TA_LIST_CONT);
DECLARE_REG_W32(ta_t *ta, STARTRENDER);

// clang-format off
AM_BEGIN(ta_t, ta_fifo_map);
  AM_RANGE(0x0000000, 0x07fffff) AM_HANDLE(NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           (w32_cb)&ta_write_poly_fifo,
                                           NULL)
  AM_RANGE(0x1000000, 0x1ffffff) AM_HANDLE(NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           (w32_cb)&ta_write_texture_fifo,
                                           NULL)
AM_END();
// clang-format on

static holly_interrupt_t list_interrupts[] = {
    HOLLY_INTC_TAEOINT,   // TA_LIST_OPAQUE
    HOLLY_INTC_TAEOMINT,  // TA_LIST_OPAQUE_MODVOL
    HOLLY_INTC_TAETINT,   // TA_LIST_TRANSLUCENT
    HOLLY_INTC_TAETMINT,  // TA_LIST_TRANSLUCENT_MODVOL
    HOLLY_INTC_TAEPTIN    // TA_LIST_PUNCH_THROUGH
};

static int param_size_lookup[0x100 * TA_NUM_PARAMS * TA_NUM_VERT_TYPES];
static int poly_type_lookup[0x100 * TA_NUM_PARAMS * TA_NUM_LISTS];
static int vertex_type_lookup[0x100 * TA_NUM_PARAMS * TA_NUM_LISTS];

static rb_callback_t ta_entry_cb = {(rb_cmp_cb)&ta_entry_cmp, NULL, NULL};

static rb_callback_t ta_context_cb = {(rb_cmp_cb)&ta_context_cmp, NULL, NULL};

static void ta_build_tables() {
  static bool initialized = false;

  if (initialized) {
    return;
  }

  initialized = true;

  for (int i = 0; i < 0x100; i++) {
    pcw_t pcw = *(pcw_t *)&i;

    for (int j = 0; j < TA_NUM_PARAMS; j++) {
      pcw.para_type = j;

      for (int k = 0; k < TA_NUM_VERT_TYPES; k++) {
        param_size_lookup[i * TA_NUM_PARAMS * TA_NUM_VERT_TYPES +
                          j * TA_NUM_VERT_TYPES + k] =
            ta_get_param_size_raw(pcw, k);
      }
    }
  }

  for (int i = 0; i < 0x100; i++) {
    pcw_t pcw = *(pcw_t *)&i;

    for (int j = 0; j < TA_NUM_PARAMS; j++) {
      pcw.para_type = j;

      for (int k = 0; k < TA_NUM_LISTS; k++) {
        pcw.list_type = k;

        poly_type_lookup[i * TA_NUM_PARAMS * TA_NUM_LISTS + j * TA_NUM_LISTS +
                         k] = ta_get_poly_type_raw(pcw);
        vertex_type_lookup[i * TA_NUM_PARAMS * TA_NUM_LISTS + j * TA_NUM_LISTS +
                           k] = ta_get_vert_type_raw(pcw);
      }
    }
  }
}

// Parameter size can be determined by only the pcw_t for every parameter other
// than vertex parameters. For vertex parameters, the vertex type derived from
// the last poly or modifier volume parameter is needed.
static int ta_get_param_size_raw(pcw_t pcw, int vertex_type) {
  switch (pcw.para_type) {
    case TA_PARAM_END_OF_LIST:
      return 32;
    case TA_PARAM_USER_TILE_CLIP:
      return 32;
    case TA_PARAM_OBJ_LIST_SET:
      return 32;
    case TA_PARAM_POLY_OR_VOL: {
      int type = ta_get_poly_type_raw(pcw);
      return type == 0 || type == 1 || type == 3 ? 32 : 64;
    }
    case TA_PARAM_SPRITE:
      return 32;
    case TA_PARAM_VERTEX: {
      return vertex_type == 0 || vertex_type == 1 || vertex_type == 2 ||
                     vertex_type == 3 || vertex_type == 4 || vertex_type == 7 ||
                     vertex_type == 8 || vertex_type == 9 || vertex_type == 10
                 ? 32
                 : 64;
    }
    default:
      return 0;
  }
}

// See "57.1.1.2 Parameter Combinations" for information on the polygon types.
int ta_get_poly_type_raw(pcw_t pcw) {
  if (pcw.list_type == TA_LIST_OPAQUE_MODVOL ||
      pcw.list_type == TA_LIST_TRANSLUCENT_MODVOL) {
    return 6;
  }

  if (pcw.para_type == TA_PARAM_SPRITE) {
    return 5;
  }

  if (pcw.volume) {
    if (pcw.col_type == 0) {
      return 3;
    }
    if (pcw.col_type == 2) {
      return 4;
    }
    if (pcw.col_type == 3) {
      return 3;
    }
  }

  if (pcw.col_type == 0 || pcw.col_type == 1 || pcw.col_type == 3) {
    return 0;
  }
  if (pcw.col_type == 2 && pcw.texture && !pcw.offset) {
    return 1;
  }
  if (pcw.col_type == 2 && pcw.texture && pcw.offset) {
    return 2;
  }
  if (pcw.col_type == 2 && !pcw.texture) {
    return 1;
  }

  return 0;
}

// See "57.1.1.2 Parameter Combinations" for information on the vertex types.
static int ta_get_vert_type_raw(pcw_t pcw) {
  if (pcw.list_type == TA_LIST_OPAQUE_MODVOL ||
      pcw.list_type == TA_LIST_TRANSLUCENT_MODVOL) {
    return 17;
  }

  if (pcw.para_type == TA_PARAM_SPRITE) {
    return pcw.texture ? 16 : 15;
  }

  if (pcw.volume) {
    if (pcw.texture) {
      if (pcw.col_type == 0) {
        return pcw.uv_16bit ? 12 : 11;
      }
      if (pcw.col_type == 2 || pcw.col_type == 3) {
        return pcw.uv_16bit ? 14 : 13;
      }
    }

    if (pcw.col_type == 0) {
      return 9;
    }
    if (pcw.col_type == 2 || pcw.col_type == 3) {
      return 10;
    }
  }

  if (pcw.texture) {
    if (pcw.col_type == 0) {
      return pcw.uv_16bit ? 4 : 3;
    }
    if (pcw.col_type == 1) {
      return pcw.uv_16bit ? 6 : 5;
    }
    if (pcw.col_type == 2 || pcw.col_type == 3) {
      return pcw.uv_16bit ? 8 : 7;
    }
  }

  if (pcw.col_type == 0) {
    return 0;
  }
  if (pcw.col_type == 1) {
    return 1;
  }
  if (pcw.col_type == 2 || pcw.col_type == 3) {
    return 2;
  }

  return 0;
}

int ta_get_param_size(pcw_t pcw, int vertex_type) {
  int size =
      param_size_lookup[pcw.obj_control * TA_NUM_PARAMS * TA_NUM_VERT_TYPES +
                        pcw.para_type * TA_NUM_VERT_TYPES + vertex_type];
  CHECK_NE(size, 0);
  return size;
}

int ta_get_poly_type(pcw_t pcw) {
  return poly_type_lookup[pcw.obj_control * TA_NUM_PARAMS * TA_NUM_LISTS +
                          pcw.para_type * TA_NUM_LISTS + pcw.list_type];
}

int ta_get_vert_type(pcw_t pcw) {
  return vertex_type_lookup[pcw.obj_control * TA_NUM_PARAMS * TA_NUM_LISTS +
                            pcw.para_type * TA_NUM_LISTS + pcw.list_type];
}

int ta_entry_cmp(const rb_node_t *lhs_it, const rb_node_t *rhs_it) {
  const texture_entry_t *lhs = rb_entry(lhs_it, const texture_entry_t, live_it);
  const texture_entry_t *rhs = rb_entry(rhs_it, const texture_entry_t, live_it);
  return lhs->key - rhs->key;
}

int ta_context_cmp(const rb_node_t *lhs_it, const rb_node_t *rhs_it) {
  const ta_ctx_t *lhs = rb_entry(lhs_it, const ta_ctx_t, live_it);
  const ta_ctx_t *rhs = rb_entry(rhs_it, const ta_ctx_t, live_it);
  return lhs->addr - rhs->addr;
}

ta_t *ta_create(struct dreamcast_s *dc, struct rb_s *rb) {
  ta_t *ta = (ta_t *)dc_create_device(dc, sizeof(ta_t), "ta",
                                      (device_init_cb)&ta_init);
  ta->base.window = window_interface_create((device_paint_cb)&ta_paint, NULL);

  ta->rb = rb;
  ta->tr = tr_create(ta->rb, ta, (get_texture_cb)&ta_get_texture);

  return ta;
}

void ta_destroy(ta_t *ta) {
  window_interface_destroy(ta->base.window);

  tr_destroy(ta->tr);

  tr_clear_context(&ta->rctx);

  // TODO cleanup rctx
  // TODO cleanup contexts

  dc_destroy_device(&ta->base);
}

bool ta_init(ta_t *ta) {
  ta_build_tables();

  ta->holly = ta->base.dc->holly;
  ta->pvr = ta->base.dc->pvr;
  ta->space = ta->base.dc->sh4->base.memory->space;
  ta->video_ram = address_space_translate(ta->space, 0x04000000);

  for (int i = 0; i < MAX_TEXTURE_ENTRIES; i++) {
    texture_entry_t *entry = &ta->entries[i];

    list_add(&ta->free_entries, &entry->free_it);
  }

  for (int i = 0; i < MAX_TILE_CONTEXTS; i++) {
    ta_ctx_t *ctx = &ta->contexts[i];

    list_add(&ta->free_contexts, &ctx->free_it);
  }

// initialize registers
#define TA_REG_R32(name)        \
  ta->pvr->reg_data[name] = ta; \
  ta->pvr->reg_read[name] = (reg_read_cb)&name##_r;
#define TA_REG_W32(name)        \
  ta->pvr->reg_data[name] = ta; \
  ta->pvr->reg_write[name] = (reg_write_cb)&name##_w;
  TA_REG_W32(SOFTRESET);
  TA_REG_W32(TA_LIST_INIT);
  TA_REG_W32(TA_LIST_CONT);
  TA_REG_W32(STARTRENDER);
#undef TA_REG_R32
#undef TA_REG_W32

  return true;
}

void ta_paint(ta_t *ta, bool show_main_menu) {
  if (ta->pending_context) {
    tr_parse_context(ta->tr, ta->pending_context, &ta->rctx, false);
    tr_render_ctx(ta->tr, &ta->rctx);

    //     // write render command after actually rendering the context so
    //     texture
    //     // insert commands will be written out first
    //     if (trace_writer_ && !last_ctx_->wrote) {
    //       trace_writer_->WriteRenderContext(last_ctx_);

    //       last_ctx_->wrote = true;
    //     }
  }

  if (show_main_menu) {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("TA")) {
        // if ((!trace_writer_ && ImGui::MenuItem("Start trace")) ||
        //     (trace_writer_ && ImGui::MenuItem("Stop trace"))) {
        //   ToggleTracing();
        // }
        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }
  }
}

texture_entry_t *ta_alloc_entry(ta_t *ta, texture_key_t key) {
  // remove from free list
  texture_entry_t *entry =
      list_first_entry(&ta->free_entries, texture_entry_t, free_it);
  CHECK_NOTNULL(entry);
  list_remove(&ta->free_entries, &entry->free_it);

  // reset entry
  memset(entry, 0, sizeof(*entry));
  entry->ta = ta;
  entry->key = key;

  // add to live tree
  rb_insert(&ta->live_entries, &entry->live_it, &ta_entry_cb);

  return entry;
}

void ta_free_entry(ta_t *ta, texture_entry_t *entry) {
  // remove from live list
  rb_unlink(&ta->live_entries, &entry->live_it, &ta_entry_cb);

  // add back to free list
  list_add(&ta->free_entries, &entry->free_it);
}

texture_handle_t ta_get_texture(ta_t *ta, const ta_ctx_t *ctx, tsp_t tsp,
                                tcw_t tcw, void *register_data,
                                register_texture_cb register_cb) {
  // clear any pending texture invalidations at this time
  ta_clear_pending_textures(ta);

  // TODO ta_ctx_t isn't considered for caching here (stride and
  // pal_pxl_format are used by TileRenderer), this feels bad
  texture_key_t texture_key = tr_get_texture_key(tsp, tcw);

  // see if an an entry already exists
  texture_entry_t search;
  search.key = texture_key;

  texture_entry_t *existing =
      rb_find_entry(&ta->live_entries, &search, live_it, &ta_entry_cb);

  if (existing) {
    return existing->handle;
  }

  // tcw_t texture_addr field is in 64-bit units
  uint32_t texture_addr = tcw.texture_addr << 3;

  // get the texture data
  uint8_t *video_ram = address_space_translate(ta->space, 0x04000000);
  uint8_t *texture = &video_ram[texture_addr];
  int width = 8 << tsp.texture_u_size;
  int height = 8 << tsp.texture_v_size;
  int element_size_bits = tcw.pixel_format == TA_PIXEL_8BPP
                              ? 8
                              : tcw.pixel_format == TA_PIXEL_4BPP ? 4 : 16;
  int texture_size = (width * height * element_size_bits) >> 3;

  // get the palette data
  uint8_t *palette_ram = address_space_translate(ta->space, 0x005f9000);
  uint8_t *palette = NULL;
  uint32_t palette_addr = 0;
  int palette_size = 0;

  if (tcw.pixel_format == TA_PIXEL_4BPP || tcw.pixel_format == TA_PIXEL_8BPP) {
    // palette ram is 4096 bytes, with each palette entry being 4 bytes each,
    // resulting in 1 << 10 indexes
    if (tcw.pixel_format == TA_PIXEL_4BPP) {
      // in 4bpp mode, the palette selector represents the upper 6 bits of the
      // palette index, with the remaining 4 bits being filled in by the texture
      palette_addr = (tcw.p.palette_selector << 4) * 4;
      palette_size = (1 << 4) * 4;
    } else if (tcw.pixel_format == TA_PIXEL_8BPP) {
      // in 4bpp mode, the palette selector represents the upper 2 bits of the
      // palette index, with the remaining 8 bits being filled in by the texture
      palette_addr = ((tcw.p.palette_selector & 0x30) << 4) * 4;
      palette_size = (1 << 8) * 4;
    }

    palette = &palette_ram[palette_addr];
  }

  // register and insert into the cache
  texture_entry_t *entry = ta_alloc_entry(ta, texture_key);

  registered_texture_t reg =
      register_cb(register_data, ctx, tsp, tcw, palette, texture);
  entry->handle = reg.handle;

  // add write callback in order to invalidate on future writes. the callback
  // address will be page aligned, therefore it will be triggered falsely in
  // some cases. over invalidate in these cases
  entry->texture_watch = add_single_write_watch(
      texture, texture_size, (memory_watch_cb)&ta_write_texture, entry);

  if (palette) {
    entry->palette_watch = add_single_write_watch(
        palette, palette_size, (memory_watch_cb)&ta_write_palette, entry);
  }

  // if (trace_writer_) {
  //   trace_writer_->WriteInsertTexture(tsp, tcw, palette, palette_size,
  //   texture,
  //                                     texture_size);
  // }

  return reg.handle;
}

void ta_clear_textures(ta_t *ta) {
  LOG_INFO("Texture cache cleared");

  rb_node_t *it = rb_first(&ta->live_entries);

  while (it) {
    rb_node_t *next = rb_next(it);

    texture_entry_t *entry = rb_entry(it, texture_entry_t, live_it);
    ta_invalidate_texture(ta, entry);

    it = next;
  }

  CHECK(!rb_first(&ta->live_entries));
}

void ta_clear_pending_textures(ta_t *ta) {
  list_for_each_entry_safe(it, &ta->invalid_entries, texture_entry_t,
                           invalid_it) {
    ta_invalidate_texture(ta, it);
    ta->num_invalidated++;
  }

  CHECK(list_empty(&ta->invalid_entries));

  prof_count("Num invalidated textures", ta->num_invalidated);
}

void ta_invalidate_texture(ta_t *ta, texture_entry_t *entry) {
  rb_free_texture(ta->rb, entry->handle);

  if (entry->texture_watch) {
    remove_memory_watch(entry->texture_watch);
  }

  if (entry->palette_watch) {
    remove_memory_watch(entry->palette_watch);
  }

  list_remove(&ta->invalid_entries, &entry->invalid_it);

  ta_free_entry(ta, entry);
}

void ta_write_texture(const re_exception_t *ex, texture_entry_t *entry) {
  ta_t *ta = entry->ta;

  // don't double remove the watch during invalidation
  entry->texture_watch = NULL;

  // add to pending invalidation list (can't remove inside of signal
  // handler)
  if (!entry->invalid_it.next) {
    list_add(&ta->invalid_entries, &entry->invalid_it);
  }
}

void ta_write_palette(const re_exception_t *ex, texture_entry_t *entry) {
  ta_t *ta = entry->ta;

  // don't double remove the watch during invalidation
  entry->palette_watch = NULL;

  // add to pending invalidation list (can't remove inside of signal
  // handler)
  if (!entry->invalid_it.next) {
    list_add(&ta->invalid_entries, &entry->invalid_it);
  }
}

void ta_soft_reset(ta_t *ta) {
  // FIXME what are we supposed to do here?
}

ta_ctx_t *ta_get_context(ta_t *ta, uint32_t addr) {
  ta_ctx_t search;
  search.addr = addr;

  ta_ctx_t *ctx =
      rb_find_entry(&ta->live_contexts, &search, live_it, &ta_context_cb);

  if (!ctx) {
    return NULL;
  }

  return ctx;
}

ta_ctx_t *ta_alloc_context(ta_t *ta, uint32_t addr) {
  // remove from free list
  ta_ctx_t *ctx = list_first_entry(&ta->free_contexts, ta_ctx_t, free_it);
  CHECK_NOTNULL(ctx);
  list_remove(&ta->free_contexts, &ctx->free_it);

  // reset it
  memset(ctx, 0, sizeof(*ctx));
  ctx->addr = addr;

  // add to live tree
  rb_insert(&ta->live_contexts, &ctx->live_it, &ta_context_cb);

  return ctx;
}

void ta_unlink_context(ta_t *ta, ta_ctx_t *ctx) {
  // remove from live tree
  rb_unlink(&ta->live_contexts, &ctx->live_it, &ta_context_cb);
}

void ta_free_context(ta_t *ta, ta_ctx_t *ctx) {
  // remove from live tree
  ta_unlink_context(ta, ctx);

  // add to free list
  list_add(&ta->free_contexts, &ctx->free_it);
}

void ta_init_context(ta_t *ta, uint32_t addr) {
  ta_ctx_t *ctx = ta_get_context(ta, addr);

  if (!ctx) {
    ctx = ta_alloc_context(ta, addr);
  }

  ctx->addr = addr;
  ctx->cursor = 0;
  ctx->size = 0;
  ctx->last_poly = NULL;
  ctx->last_vertex = NULL;
  ctx->list_type = 0;
  ctx->vertex_type = 0;
}

void ta_write_context(ta_t *ta, uint32_t addr, uint32_t value) {
  ta_ctx_t *ctx = ta_get_context(ta, addr);
  CHECK_NOTNULL(ctx);

  CHECK_LT(ctx->size + 4, (int)sizeof(ctx->data));
  *(uint32_t *)&ctx->data[ctx->size] = value;
  ctx->size += 4;

  // each TA command is either 32 or 64 bytes, with the pcw_t being in the first
  // 32 bytes always. check every 32 bytes to see if the command has been
  // completely received or not
  if (ctx->size % 32 == 0) {
    void *data = &ctx->data[ctx->cursor];
    pcw_t pcw = *(pcw_t *)data;

    int size = ta_get_param_size(pcw, ctx->vertex_type);
    int recv = ctx->size - ctx->cursor;

    if (recv < size) {
      // wait for the entire command
      return;
    }

    if (pcw.para_type == TA_PARAM_END_OF_LIST) {
      holly_raise_interrupt(ta->holly, list_interrupts[ctx->list_type]);

      ctx->last_poly = NULL;
      ctx->last_vertex = NULL;
      ctx->list_type = 0;
      ctx->vertex_type = 0;
    } else if (pcw.para_type == TA_PARAM_OBJ_LIST_SET) {
      LOG_FATAL("TA_PARAM_OBJ_LIST_SET unsupported");
    } else if (pcw.para_type == TA_PARAM_POLY_OR_VOL) {
      ctx->last_poly = (poly_param_t *)data;
      ctx->last_vertex = NULL;
      ctx->list_type = ctx->last_poly->type0.pcw.list_type;
      ctx->vertex_type = ta_get_vert_type(ctx->last_poly->type0.pcw);
    } else if (pcw.para_type == TA_PARAM_SPRITE) {
      ctx->last_poly = (poly_param_t *)data;
      ctx->last_vertex = NULL;
      ctx->list_type = ctx->last_poly->type0.pcw.list_type;
      ctx->vertex_type = ta_get_vert_type(ctx->last_poly->type0.pcw);
    }

    ctx->cursor += recv;
  }
}

void ta_finish_context(ta_t *ta, uint32_t addr) {
  ta_ctx_t *ctx = ta_get_context(ta, addr);
  CHECK_NOTNULL(ctx);

  // save required register state being that the actual rendering of this
  // context will be deferred
  ta_save_state(ta, ctx);

  // tell holly that rendering is complete
  holly_raise_interrupt(ta->holly, HOLLY_INTC_PCEOVINT);
  holly_raise_interrupt(ta->holly, HOLLY_INTC_PCEOIINT);
  holly_raise_interrupt(ta->holly, HOLLY_INTC_PCEOTINT);

  // free the last pending context
  if (ta->pending_context) {
    ta_free_context(ta, ta->pending_context);
    ta->pending_context = NULL;
  }

  // set this context to pending
  ta_unlink_context(ta, ctx);

  ta->pending_context = ctx;
}

void ta_save_state(ta_t *ta, ta_ctx_t *ctx) {
  pvr_t *pvr = ta->pvr;

  // autosort
  if (!pvr->FPU_PARAM_CFG->region_header_type) {
    ctx->autosort = !pvr->ISP_FEED_CFG->presort;
  } else {
    uint32_t region_data =
        address_space_r32(ta->space, 0x05000000 + *pvr->REGION_BASE);
    ctx->autosort = !(region_data & 0x20000000);
  }

  // texture stride
  ctx->stride = pvr->TEXT_CONTROL->stride * 32;

  // texture palette pixel format
  ctx->pal_pxl_format = pvr->PAL_RAM_CTRL->pixel_format;

  // write out video width to help with unprojecting the screen space
  // coordinates
  if (pvr->SPG_CONTROL->interlace ||
      (!pvr->SPG_CONTROL->NTSC && !pvr->SPG_CONTROL->PAL)) {
    // interlaced and VGA mode both render at full resolution
    ctx->video_width = 640;
    ctx->video_height = 480;
  } else {
    ctx->video_width = 320;
    ctx->video_height = 240;
  }

  // according to the hardware docs, this is the correct calculation of the
  // background ISP address. however, in practice, the second TA buffer's ISP
  // address comes out to be 0x800000 when booting the bios and the vram is
  // only 8mb total. by examining a raw memory dump, the ISP data is only ever
  // available at 0x0 when booting the bios, so masking this seems to be the
  // correct solution
  uint32_t vram_offset =
      0x05000000 +
      ((ctx->addr + pvr->ISP_BACKGND_T->tag_address * 4) & 0x7fffff);

  // get surface parameters
  ctx->bg_isp.full = address_space_r32(ta->space, vram_offset);
  ctx->bg_tsp.full = address_space_r32(ta->space, vram_offset + 4);
  ctx->bg_tcw.full = address_space_r32(ta->space, vram_offset + 8);
  vram_offset += 12;

  // get the background depth
  ctx->bg_depth = *(float *)pvr->ISP_BACKGND_D;

  // get the byte size for each vertex. normally, the byte size is
  // ISP_BACKGND_T.skip + 3, but if parameter selection volume mode is in
  // effect and the shadow bit is 1, then the byte size is
  // ISP_BACKGND_T.skip * 2 + 3
  int vertex_size = pvr->ISP_BACKGND_T->skip;
  if (!pvr->FPU_SHAD_SCALE->intensity_volume_mode &&
      pvr->ISP_BACKGND_T->shadow) {
    vertex_size *= 2;
  }
  vertex_size = (vertex_size + 3) * 4;

  // skip to the first vertex
  vram_offset += pvr->ISP_BACKGND_T->tag_offset * vertex_size;

  // copy vertex data to context
  for (int i = 0, bg_offset = 0; i < 3; i++) {
    CHECK_LE(bg_offset + vertex_size, (int)sizeof(ctx->bg_vertices));

    address_space_memcpy_to_host(ta->space, &ctx->bg_vertices[bg_offset],
                                 vram_offset, vertex_size);

    bg_offset += vertex_size;
    vram_offset += vertex_size;
  }
}

// TODO
// void ta_toggle_tracing(ta_t *ta) {
//   if (!trace_writer_) {
//     char filename[PATH_MAX];
//     GetNextTraceFilename(filename, sizeof(filename));

//     trace_writer_ = new TraceWriter();

//     if (!trace_writer_->Open(filename)) {
//       delete trace_writer_;
//       trace_writer_ = NULL;

//       LOG_INFO("Failed to start tracing");

//       return;
//     }

//     // clear texture cache in order to generate insert events for all
//     textures
//     // referenced while tracing
//     ClearTextures();

//     LOG_INFO("Begin tracing to %s", filename);
//   } else {
//     delete trace_writer_;
//     trace_writer_ = NULL;

//     LOG_INFO("End tracing");
//   }
// }

void ta_write_poly_fifo(ta_t *ta, uint32_t addr, uint32_t value) {
  ta_write_context(ta, ta->pvr->TA_ISP_BASE->base_address, value);
}

void ta_write_texture_fifo(ta_t *ta, uint32_t addr, uint32_t value) {
  addr &= 0xeeffffff;
  *(uint32_t *)&ta->video_ram[addr] = value;
}

REG_W32(ta_t *ta, SOFTRESET) {
  if (!(*new_value & 0x1)) {
    return;
  }

  ta_soft_reset(ta);
}

REG_W32(ta_t *ta, TA_LIST_INIT) {
  if (!(*new_value & 0x80000000)) {
    return;
  }

  ta_init_context(ta, ta->pvr->TA_ISP_BASE->base_address);
}

REG_W32(ta_t *ta, TA_LIST_CONT) {
  if (!(*new_value & 0x80000000)) {
    return;
  }

  LOG_WARNING("Unsupported TA_LIST_CONT");
}

REG_W32(ta_t *ta, STARTRENDER) {
  if (!*new_value) {
    return;
  }

  ta_finish_context(ta, ta->pvr->PARAM_BASE->base_address);
}