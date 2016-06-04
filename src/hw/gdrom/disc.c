#include "core/assert.h"
#include "hw/gdrom/disc.h"

static const int GDI_PREGAP_SIZE = 150;
static const int GDI_MAX_TRACKS = 64;

typedef struct disc_s {
  void (*destroy)(struct disc_s *);
  int (*num_tracks)(struct disc_s *);
  track_t *(*get_track)(struct disc_s *, int);
  int (*read_sector)(struct disc_s *, int, void *);
} disc_t;

typedef struct {
  disc_t base;

  track_t tracks[GDI_MAX_TRACKS];
  int num_tracks;
} gdi_t;

static gdi_t *gdi_create(const char *filename);
static void gdi_destroy(gdi_t *gdi);
static int gdi_num_tracks(gdi_t *gdi);
static track_t *gdi_get_track(gdi_t *gdi, int n);
static int gdi_read_sector(gdi_t *gdi, int fad, void *dst);

gdi_t *gdi_create(const char *filename) {
  gdi_t *gdi = calloc(1, sizeof(gdi_t));

  gdi->base.destroy = (void (*)(struct disc_s *)) & gdi_destroy;
  gdi->base.num_tracks = (int (*)(struct disc_s *)) & gdi_num_tracks;
  gdi->base.get_track = (track_t * (*)(struct disc_s *, int)) & gdi_get_track;
  gdi->base.read_sector =
      (int (*)(struct disc_s *, int, void *)) & gdi_read_sector;

  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    free(gdi);
    return NULL;
  }

  // <number of tracks>
  // <track num> <lba> <ctrl> <sector size> <file name> <file offset>
  // <track num> <lba> <ctrl> <sector size> <file name> <file offset>
  int num_tracks;
  int n = fscanf(fp, "%d", &num_tracks);
  if (n != 1) {
    free(gdi);
    fclose(fp);
    return NULL;
  }

  // get gdi dirname to help resolve track paths
  char dirname[PATH_MAX];
  fs_dirname(filename, dirname, sizeof(dirname));

  for (int i = 0; i < num_tracks; i++) {
    int num, lba, ctrl, sector_size, file_offset;
    char filename[PATH_MAX];
    n = fscanf(fp, "%d %d %d %d %s %d", &num, &lba, &ctrl, &sector_size,
               filename, &file_offset);
    if (n != 6) {
      free(gdi);
      fclose(fp);
      return NULL;
    }

    // add track
    CHECK_LT(gdi->num_tracks, GDI_MAX_TRACKS);
    track_t *track = &gdi->tracks[gdi->num_tracks++];
    track->num = num;
    track->fad = lba + GDI_PREGAP_SIZE;
    track->ctrl = ctrl;
    track->file_offset = file_offset - track->fad * SECTOR_SIZE;
    snprintf(track->filename, sizeof(track->filename), "%s" PATH_SEPARATOR "%s",
             dirname, filename);
  }

  fclose(fp);

  return gdi;
}

void gdi_destroy(gdi_t *gdi) {
  // cleanup file handles
  for (int i = 0; i < gdi->num_tracks; i++) {
    track_t *track = &gdi->tracks[i];

    if (track->file) {
      fclose(track->file);
    }
  }
}

int gdi_num_tracks(gdi_t *gdi) {
  return gdi->num_tracks;
}

track_t *gdi_get_track(gdi_t *gdi, int n) {
  return &gdi->tracks[n];
}

int gdi_read_sector(gdi_t *gdi, int fad, void *dst) {
  // find the track to read from
  track_t *track = NULL;
  for (int i = 0; i < gdi->num_tracks; i++) {
    track_t *curr_track = &gdi->tracks[i];
    track_t *next_track = i < gdi->num_tracks - 1 ? &gdi->tracks[i + 1] : NULL;

    if (fad >= curr_track->fad && (!next_track || fad < next_track->fad)) {
      track = curr_track;
      break;
    }
  }
  CHECK_NOTNULL(track);

  // open the file backing the track
  if (!track->file) {
    track->file = fopen(track->filename, "rb");
    CHECK(track->file);
  }

  // read from it
  int res =
      fseek(track->file, track->file_offset + fad * SECTOR_SIZE, SEEK_SET);
  CHECK_EQ(res, 0);

  res = (int)fread(dst, SECTOR_SIZE, 1, track->file);
  CHECK_EQ(res, 1);

  return 1;
}

struct disc_s *disc_create_gdi(const char *filename) {
  return (struct disc_s *)gdi_create(filename);
}

void disc_destroy(struct disc_s *disc) {
  return disc->destroy(disc);
}

int disc_num_tracks(struct disc_s *disc) {
  return disc->num_tracks(disc);
}

track_t *disc_get_track(struct disc_s *disc, int n) {
  return disc->get_track(disc, n);
}

int disc_read_sector(struct disc_s *disc, int fad, void *dst) {
  return disc->read_sector(disc, fad, dst);
}