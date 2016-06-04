#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdbool.h>
#include <string.h>
#include "core/constructor.h"
#include "core/list.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_OPTION_BOOL(name) extern bool OPTION_##name;

#define DEFINE_OPTION_BOOL(name, value, desc)     \
  bool OPTION_##name;                             \
  static option_t OPTION_T_##name = {             \
      OPT_BOOL, #name, desc, &OPTION_##name, {}}; \
  CONSTRUCTOR(OPTION_REGISTER_##name) {           \
    *(bool *)(&OPTION_##name) = value;            \
    option_register(&OPTION_T_##name);            \
  }                                               \
  DESTRUCTOR(OPTION_UNREGISTER_##name) {          \
    option_unregister(&OPTION_T_##name);          \
  }

#define DECLARE_OPTION_INT(name) extern int OPTION_##name;

#define DEFINE_OPTION_INT(name, value, desc)     \
  int OPTION_##name;                             \
  static option_t OPTION_T_##name = {            \
      OPT_INT, #name, desc, &OPTION_##name, {}}; \
  CONSTRUCTOR(OPTION_REGISTER_##name) {          \
    *(int *)(&OPTION_##name) = value;            \
    option_register(&OPTION_T_##name);           \
  }                                              \
  DESTRUCTOR(OPTION_UNREGISTER_##name) {         \
    option_unregister(&OPTION_T_##name);         \
  }

#define DECLARE_OPTION_STRING(name) extern char OPTION_##name[1024];

#define DEFINE_OPTION_STRING(name, value, desc)     \
  char OPTION_##name[1024];                         \
  static option_t OPTION_T_##name = {               \
      OPT_STRING, #name, desc, &OPTION_##name, {}}; \
  CONSTRUCTOR(OPTION_REGISTER_##name) {             \
    strcpy((char *) & OPTION_##name, value);        \
    option_register(&OPTION_T_##name);              \
  }                                                 \
  DESTRUCTOR(OPTION_UNREGISTER_##name) {            \
    option_unregister(&OPTION_T_##name);            \
  }

typedef enum {
  OPT_BOOL,
  OPT_INT,
  OPT_STRING,
} option_type_t;

typedef struct option_s {
  option_type_t type;
  const char *name;
  const char *desc;
  void *storage;
  list_node_t it;
} option_t;

void option_register(option_t *option);
void option_unregister(option_t *option);

void option_parse(int *argc, char ***argv);
void option_print_help();

#ifdef __cplusplus
}
#endif

#endif