#ifndef UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
#define UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015

#ifdef __cplusplus
#include <cstdint>        // for uint64_t
#include <cstdio>
#include <ctime>
#else
#include <stdbool.h>
#include <stdint.h>       // for uint64_t
#include <stdio.h>
#include <time.h>
#endif

#include <cuda_runtime.h>
#include <nppdefs.h>      // for NppStatus
#include <npp.h> // NPP_VERSION_MINOR

#include "defs.h"

#if NPP_VERSION_MAJOR >= 12
#define NPP_NEW_API 1
#define NPP_CONTEXTIZE(fn_name) fn_name##_Ctx
#else
#define NPP_CONTEXTIZE(...) __VA_ARGS__
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

#define GPU_TIMER_START(name, req_ll, stream)                                  \
        int gpu_timer_##name##_enabled = log_level >= req_ll;                  \
        cudaStream_t gpu_timer_##name##_stream = stream;                       \
        if (gpu_timer_##name##_enabled) {                                      \
                cudaEventRecord(cuda_event_start, stream);                     \
        }

#define GPU_TIMER_STOP(name)                                                   \
        if (gpu_timer_##name##_enabled) {                                      \
                cudaEventRecord(cuda_event_stop, gpu_timer_##name##_stream);   \
                float elapsedTimeMs = 0;                                       \
                cudaEventSynchronize(cuda_event_stop);                         \
                cudaEventElapsedTime(&elapsedTimeMs, cuda_event_start,         \
                                     cuda_event_stop);                         \
                fprintf(stderr, #name " duration %f s\n",                      \
                        elapsedTimeMs / 1000.0);                               \
        }

extern const char *fg_bold;
extern const char *fg_red;
extern const char *fg_yellow;
extern const char *term_reset;
extern cudaEvent_t cuda_event_start;
extern cudaEvent_t cuda_event_stop;

#if __STDC_VERSION__ >= 202311L || __cplusplus >= 202002L || __GNUC__ >= 12 || \
    __clang_major__ >= 9
#define ERROR_MSG(fmt, ...)                                                    \
        fprintf(stderr, "%s" fmt "%s", fg_red __VA_OPT__(, ) __VA_ARGS__,      \
                term_reset)
#define WARN_MSG(fmt, ...)                                                     \
        if (log_level != LL_REALLY_QUIET)                                      \
        fprintf(stderr, "%s" fmt "%s", fg_yellow __VA_OPT__(, ) __VA_ARGS__,   \
                term_reset)
#else
#define ERROR_MSG(...) fprintf(stderr, __VA_ARGS__)
#define WARN_MSG(...)                                                          \
        if (log_level != LL_REALLY_QUIET)                                      \
        fprintf(stderr, __VA_ARGS__)
#endif

// DHR doesn't pass -QQ but just -Q so this is a workaround
#ifdef INSIDE_DHR
#undef WARN_MSG
#define WARN_MSG(...)
#endif

#define INFO_MSG(...)                                                          \
        if (log_level >= LL_INFO)                                              \
        fprintf(stderr, __VA_ARGS__)
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

#define CHECK_CUDA(call)                                                       \
        {                                                                      \
                cudaError_t err = call;                                        \
                if (cudaSuccess != err) {                                      \
                        ERROR_MSG(                                             \
                            "Cuda error in file '%s' in line %i : %s.\n",      \
                            __FILE__, __LINE__, cudaGetErrorString(err));      \
                        abort();                                               \
                }                                                              \
        }
#define CHECK_NPP(call)                                                        \
        {                                                                      \
                NppStatus rc = call;                                           \
                if (NPP_NO_ERROR != rc) {                                      \
                        ERROR_MSG(                                             \
                            "NPP error in file '%s' in line %i : %s (%d).\n",  \
                            __FILE__, __LINE__, npp_status_to_str(rc),         \
                            (int)rc);                                          \
                        abort();                                               \
                }                                                              \
        }


#ifdef __cplusplus
extern "C" {
#endif // _cplusplus

const char *npp_status_to_str(NppStatus rc);
const char *nvtiff_status_to_str(int rc);
const char *nvj2k_status_to_str(int rc);
enum {
        UINT64_ASCII_LEN = 20,
};
char* format_number_with_delim(uint64_t num, char* buf, size_t buflen);
size_t get_cuda_dev_global_memory();

extern const char *const coord_pos_name[4];

#ifdef NPP_NEW_API
void init_npp_context(NppStreamContext *nppStreamCtx,
                               cudaStream_t stream);
#endif

struct dec_image;
struct owned_image;
struct owned_image *new_cuda_owned_image(const struct dec_image *in);
// struct owned_image *new_cuda_owned_float_image(const struct dec_image *in);
struct owned_image *copy_img_from_device(const struct dec_image *in,
                                         enum out_format output_format,
                                         cudaStream_t stream);

void gcs_to_webm(double latitude, double longitude, double *y, double *x);

bool gj_adjust_size(int *width, int *height, int comp_count);

struct tie_points tuple6_to_tie_points(unsigned count,
                                       const double *tie_points);

enum nd_feature get_nd_feature(const char *filename1, const char *filename2);

#ifdef __cplusplus
}
#endif // _cplusplus

#endif // defined UTILS_H_3A62EF66_2DE8_441D_8381_B3FBB49EC015
