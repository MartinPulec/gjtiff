#ifndef UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
#define UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015

#ifdef __cplusplus
#include <cstdio>
#include <ctime>
#else
#include <time.h>
#include <stdio.h>
#endif

#include <cuda_runtime.h>

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

#define FG_RED "\033[31m"
#define FG_YELLOW "\033[33m"
#define TERM_RESET "\033[0m"
#if __cplusplus > 201703L
#define ERROR_MSG(fmt, ...)                                                    \
        fprintf(stderr, FG_RED fmt TERM_RESET __VA_OPT__(, ) __VA_ARGS__)
#define WARN_MSG(fmt, ...)                                                    \
        fprintf(stderr, FG_YELLOW fmt TERM_RESET __VA_OPT__(, ) __VA_ARGS__)
#else
#define ERROR_MSG(...) fprintf(stderr, __VA_ARGS__)
#define WARN_MSG(...) fprintf(stderr, __VA_ARGS__)
#endif

#if CUDART_VERSION <= 8000
#define cudaFreeAsync(ptr, stream)                                             \
        cudaStreamSynchronize(stream);                                         \
        cudaFree(ptr)
#define cudaMallocAsync(ptr, size, stream)                                     \
        cudaStreamSynchronize(stream);                                         \
        cudaMalloc(ptr, size)
#endif

#endif // defined UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
