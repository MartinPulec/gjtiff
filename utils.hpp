#include <ctime>

#define TIMER_DECLARE(name) struct timespec t0_##name, t1_##name
#define TIMER_START(name, log_level)                                           \
  TIMER_DECLARE(name);                                                         \
  if (log_level >= 2)                                                          \
  timespec_get(&t0_##name, TIME_UTC)
#define TIMER_STOP(name, log_level)                                            \
  if (log_level >= 2) {                                                        \
    timespec_get(&t1_##name, TIME_UTC);                                        \
    fprintf(stderr, #name " duration %f s\n",                                  \
            t1_##name.tv_sec - t0_##name.tv_sec +                              \
                (t1_##name.tv_nsec - t0_##name.tv_nsec) / 1000000000.0);       \
  }
