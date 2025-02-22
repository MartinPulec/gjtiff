#ifndef UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
#define UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015

#ifdef __cplusplus
#include <cstdint>        // for uint64_t
#include <cstdio>
#include <ctime>
#else
#include <stdint.h>       // for uint64_t
#include <stdio.h>
#include <time.h>
#endif

#include <cuda_runtime.h>
#include <nppdefs.h>      // for NppStatus

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define TIMER_DECLARE(name, enabled)                                           \
        struct timespec t0_##name = {0, 0};                                    \
        struct timespec t1_##name = {0, 0};                                    \
        int name##_enabled = enabled;
#define TIMER_START(name, req_ll)                                              \
        TIMER_DECLARE(name, log_level >= req_ll);                              \
        if (name##_enabled)                                                     \
                timespec_get(&t0_##name, TIME_UTC)
#define TIMER_STOP(name)                                                       \
        if (name##_enabled) {                                                   \
                timespec_get(&t1_##name, TIME_UTC);                            \
                fprintf(stderr, #name " duration %f s\n",                      \
                        t1_##name.tv_sec - t0_##name.tv_sec +                  \
                            (t1_##name.tv_nsec - t0_##name.tv_nsec) /          \
                                1000000000.0);                                 \
        }

extern const char *fg_bold;
extern const char *fg_red;
extern const char *fg_yellow;
extern const char *term_reset;

#if __STDC_VERSION__ >= 202311L || __cplusplus >= 202002L || __GNUC__ >= 12 || \
    __clang_major__ >= 9
#define ERROR_MSG(fmt, ...)                                                    \
        fprintf(stderr, "%s" fmt "%s", fg_red __VA_OPT__(, ) __VA_ARGS__,      \
                term_reset)
#define WARN_MSG(fmt, ...)                                                     \
        fprintf(stderr, "%s" fmt "%s", fg_yellow __VA_OPT__(, ) __VA_ARGS__,   \
                term_reset)
#else
#define ERROR_MSG(...) fprintf(stderr, __VA_ARGS__)
#define WARN_MSG(...) fprintf(stderr, __VA_ARGS__)
#endif

#define VERBOSE_MSG(...)                                                       \
        if (log_level >= LL_VERBOSE)                                           \
        fprintf(stderr, __VA_ARGS__)
#define DEBUG_MSG(...)                                                         \
        if (log_level >= LL_DEBUG)                                             \
        fprintf(stderr, __VA_ARGS__)

#if CUDART_VERSION <= 8000
#define cudaFreeAsync(ptr, stream)                                             \
        cudaStreamSynchronize(stream);                                         \
        cudaFree(ptr)
#define cudaMallocAsync(ptr, size, stream)                                     \
        cudaStreamSynchronize(stream);                                         \
        cudaMalloc(ptr, size)
#endif

EXTERN_C const char *npp_status_to_str(NppStatus rc);
EXTERN_C const char *nvtiff_status_to_str(int rc);
EXTERN_C const char *nvj2k_status_to_str(int rc);
enum {
        UINT64_ASCII_LEN = 20,
};
EXTERN_C char* format_number_with_delim(uint64_t num, char* buf, size_t buflen);

extern const char *const coord_pos_name[4];

#endif // defined UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
