#ifndef AICA_H
#define AICA_H

#include <stdint.h>
#include "hw/memory.h"

struct aica;
struct dreamcast;

#define AICA_SAMPLE_FREQ 44100

AM_DECLARE(aica_reg_map);
AM_DECLARE(aica_data_map);

struct aica *aica_create(struct dreamcast *dc);
void aica_destroy(struct aica *aica);

int aica_available_frames(struct aica *aica);
int aica_read_frames(struct aica *aica, void *buffer, int num_frames);

#endif
