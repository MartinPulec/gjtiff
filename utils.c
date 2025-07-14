#include "utils.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdlib.h>        // for abort
#include <string.h>        // for memset

#include "defs.h" // for ARR_SIZE

#define STATUS_TO_NAME(x) {x,#x}

static const struct npp_status_map_t {
        NppStatus rc;
        const char *name;
} npp_status_mapping[] = {
        STATUS_TO_NAME(NPP_NOT_SUPPORTED_MODE_ERROR),
        STATUS_TO_NAME(NPP_INVALID_HOST_POINTER_ERROR),
        STATUS_TO_NAME(NPP_INVALID_DEVICE_POINTER_ERROR),
        STATUS_TO_NAME(NPP_LUT_PALETTE_BITSIZE_ERROR),
        STATUS_TO_NAME(NPP_ZC_MODE_NOT_SUPPORTED_ERROR),
        STATUS_TO_NAME(NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY),
        STATUS_TO_NAME(NPP_TEXTURE_BIND_ERROR),
        STATUS_TO_NAME(NPP_WRONG_INTERSECTION_ROI_ERROR),
        STATUS_TO_NAME(NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR),
        STATUS_TO_NAME(NPP_MEMFREE_ERROR),
        STATUS_TO_NAME(NPP_MEMSET_ERROR),
        STATUS_TO_NAME(NPP_MEMCPY_ERROR),
        STATUS_TO_NAME(NPP_ALIGNMENT_ERROR),
        STATUS_TO_NAME(NPP_CUDA_KERNEL_EXECUTION_ERROR),
        {-215, "NPP_STREAM_CTX_ERROR"},
        STATUS_TO_NAME(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR),
        STATUS_TO_NAME(NPP_QUALITY_INDEX_ERROR),
        STATUS_TO_NAME(NPP_RESIZE_NO_OPERATION_ERROR),
        STATUS_TO_NAME(NPP_OVERFLOW_ERROR),
        STATUS_TO_NAME(NPP_NOT_EVEN_STEP_ERROR),
        STATUS_TO_NAME(NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR),
        STATUS_TO_NAME(NPP_LUT_NUMBER_OF_LEVELS_ERROR),
        STATUS_TO_NAME(NPP_CORRUPTED_DATA_ERROR),
        STATUS_TO_NAME(NPP_CHANNEL_ORDER_ERROR),
        STATUS_TO_NAME(NPP_ZERO_MASK_VALUE_ERROR),
        STATUS_TO_NAME(NPP_QUADRANGLE_ERROR),
        STATUS_TO_NAME(NPP_RECTANGLE_ERROR),
        STATUS_TO_NAME(NPP_COEFFICIENT_ERROR),
        STATUS_TO_NAME(NPP_NUMBER_OF_CHANNELS_ERROR),
        STATUS_TO_NAME(NPP_COI_ERROR),
        STATUS_TO_NAME(NPP_DIVISOR_ERROR),
        STATUS_TO_NAME(NPP_CHANNEL_ERROR),
        STATUS_TO_NAME(NPP_STRIDE_ERROR),
        STATUS_TO_NAME(NPP_ANCHOR_ERROR),
        STATUS_TO_NAME(NPP_MASK_SIZE_ERROR),
        STATUS_TO_NAME(NPP_RESIZE_FACTOR_ERROR),
        STATUS_TO_NAME(NPP_INTERPOLATION_ERROR),
        STATUS_TO_NAME(NPP_MIRROR_FLIP_ERROR),
        STATUS_TO_NAME(NPP_MOMENT_00_ZERO_ERROR),
        STATUS_TO_NAME(NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR),
        STATUS_TO_NAME(NPP_THRESHOLD_ERROR),
        STATUS_TO_NAME(NPP_CONTEXT_MATCH_ERROR),
        STATUS_TO_NAME(NPP_FFT_FLAG_ERROR),
        STATUS_TO_NAME(NPP_FFT_ORDER_ERROR),
        STATUS_TO_NAME(NPP_STEP_ERROR),
        STATUS_TO_NAME(NPP_SCALE_RANGE_ERROR),
        STATUS_TO_NAME(NPP_DATA_TYPE_ERROR),
        STATUS_TO_NAME(NPP_OUT_OFF_RANGE_ERROR),
        STATUS_TO_NAME(NPP_DIVIDE_BY_ZERO_ERROR),
        STATUS_TO_NAME(NPP_MEMORY_ALLOCATION_ERR),
        STATUS_TO_NAME(NPP_NULL_POINTER_ERROR),
        STATUS_TO_NAME(NPP_RANGE_ERROR),
        STATUS_TO_NAME(NPP_SIZE_ERROR),
        STATUS_TO_NAME(NPP_BAD_ARGUMENT_ERROR),
        STATUS_TO_NAME(NPP_NO_MEMORY_ERROR),
        STATUS_TO_NAME(NPP_NOT_IMPLEMENTED_ERROR),
        STATUS_TO_NAME(NPP_ERROR),
        STATUS_TO_NAME(NPP_ERROR_RESERVED),
        STATUS_TO_NAME(NPP_NO_ERROR),
#ifdef NPP_NEW_API
        STATUS_TO_NAME(NPP_SUCCESS),
#endif
        STATUS_TO_NAME(NPP_NO_OPERATION_WARNING),
        STATUS_TO_NAME(NPP_DIVIDE_BY_ZERO_WARNING),
        STATUS_TO_NAME(NPP_AFFINE_QUAD_INCORRECT_WARNING),
        STATUS_TO_NAME(NPP_WRONG_INTERSECTION_ROI_WARNING),
        STATUS_TO_NAME(NPP_WRONG_INTERSECTION_QUAD_WARNING),
        STATUS_TO_NAME(NPP_DOUBLE_SIZE_WARNING),
        STATUS_TO_NAME(NPP_MISALIGNED_DST_ROI_WARNING),
};

const char *npp_status_to_str(NppStatus rc) {
        for (unsigned i = 0; i < ARR_SIZE(npp_status_mapping); ++i) {
                if (npp_status_mapping[i].rc == rc) {
                        return npp_status_mapping[i].name;
                }
        }
        return "(unknown)";
}

typedef enum {
        NVTIFF_STATUS_SUCCESS = 0,
        NVTIFF_STATUS_NOT_INITIALIZED = 1,
        NVTIFF_STATUS_INVALID_PARAMETER = 2,
        NVTIFF_STATUS_BAD_TIFF = 3,
        NVTIFF_STATUS_TIFF_NOT_SUPPORTED = 4,
        NVTIFF_STATUS_ALLOCATOR_FAILURE = 5,
        NVTIFF_STATUS_EXECUTION_FAILED = 6,
        NVTIFF_STATUS_ARCH_MISMATCH = 7,
        NVTIFF_STATUS_INTERNAL_ERROR = 8,
        NVTIFF_STATUS_NVCOMP_NOT_FOUND = 9,
        NVTIFF_STATUS_NVJPEG_NOT_FOUND = 10,
        NVTIFF_STATUS_TAG_NOT_FOUND = 11,
        NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS = 12,
} nvtiffStatus_t;

const char *nvtiff_status_to_str(int rc)
{
        static const struct npp_status_map_t {
                int rc;
                const char *name;
        } nvtiff_status_mapping[] = {
            STATUS_TO_NAME(NVTIFF_STATUS_SUCCESS),
            STATUS_TO_NAME(NVTIFF_STATUS_NOT_INITIALIZED),
            STATUS_TO_NAME(NVTIFF_STATUS_INVALID_PARAMETER),
            STATUS_TO_NAME(NVTIFF_STATUS_BAD_TIFF),
            STATUS_TO_NAME(NVTIFF_STATUS_TIFF_NOT_SUPPORTED),
            STATUS_TO_NAME(NVTIFF_STATUS_ALLOCATOR_FAILURE),
            STATUS_TO_NAME(NVTIFF_STATUS_EXECUTION_FAILED),
            STATUS_TO_NAME(NVTIFF_STATUS_ARCH_MISMATCH),
            STATUS_TO_NAME(NVTIFF_STATUS_INTERNAL_ERROR),
            STATUS_TO_NAME(NVTIFF_STATUS_NVCOMP_NOT_FOUND),
            STATUS_TO_NAME(NVTIFF_STATUS_NVJPEG_NOT_FOUND),
            STATUS_TO_NAME(NVTIFF_STATUS_TAG_NOT_FOUND),
            STATUS_TO_NAME(NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS),
        };
        for (unsigned i = 0; i < ARR_SIZE(nvtiff_status_mapping); ++i) {
                if (nvtiff_status_mapping[i].rc == rc) {
                        return nvtiff_status_mapping[i].name;
                }
        }
        return "(unknown)";
}

const char *nvj2k_status_to_str(int rc)
{
        enum {
                NVJPEG2K_STATUS_SUCCESS = 0,
                NVJPEG2K_STATUS_NOT_INITIALIZED = 1,
                NVJPEG2K_STATUS_INVALID_PARAMETER = 2,
                NVJPEG2K_STATUS_BAD_JPEG = 3,
                NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED = 4,
                NVJPEG2K_STATUS_ALLOCATOR_FAILURE = 5,
                NVJPEG2K_STATUS_EXECUTION_FAILED = 6,
                NVJPEG2K_STATUS_ARCH_MISMATCH = 7,
                NVJPEG2K_STATUS_INTERNAL_ERROR = 8,
                NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
        };
        static const struct npp_status_map_t {
                int rc;
                const char *name;
        } nvj2k_status_mapping[] = {
            STATUS_TO_NAME(NVJPEG2K_STATUS_SUCCESS),
            STATUS_TO_NAME(NVJPEG2K_STATUS_NOT_INITIALIZED),
            STATUS_TO_NAME(NVJPEG2K_STATUS_INVALID_PARAMETER),
            STATUS_TO_NAME(NVJPEG2K_STATUS_BAD_JPEG),
            STATUS_TO_NAME(NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED),
            STATUS_TO_NAME(NVJPEG2K_STATUS_ALLOCATOR_FAILURE),
            STATUS_TO_NAME(NVJPEG2K_STATUS_EXECUTION_FAILED),
            STATUS_TO_NAME(NVJPEG2K_STATUS_ARCH_MISMATCH),
            STATUS_TO_NAME(NVJPEG2K_STATUS_INTERNAL_ERROR),
            STATUS_TO_NAME(NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED),
        };
        for (unsigned i = 0; i < ARR_SIZE(nvj2k_status_mapping); ++i) {
                if (nvj2k_status_mapping[i].rc == rc) {
                        return nvj2k_status_mapping[i].name;
                }
        }
        return "(unknown)";
}

const char *const coord_pos_name[4] = {"Upper Left", "Upper Right",
                                       "Lower Right", "Lower Left"};

/// format number with thousands delimited by ','
char *format_number_with_delim(uint64_t num, char *buf, size_t buflen)
{
        assert(buflen >= 1);
        buf[buflen - 1] = '\0';
        char *ptr = buf + buflen - 1;
        int grp_count = 0;
        do {
                if (ptr == buf || (grp_count == 3 && ptr == buf + 1)) {
                        snprintf(buf, buflen, "%s", "ERR");
                        return buf;
                }
                if (grp_count++ == 3) {
                        grp_count = 1;
                        *--ptr = ',';
                }
                *--ptr = (char)('0' + (num % 10));
                num /= 10;
        } while (num != 0);

        return ptr;
}

size_t get_cuda_dev_global_memory()
{

        int cur_device = 0;
        CHECK_CUDA(cudaGetDevice(&cur_device));
        struct cudaDeviceProp device_properties;
        CHECK_CUDA(cudaGetDeviceProperties(&device_properties, cur_device));

        VERBOSE_MSG("CUDA device %d total memory %zd GB\n", cur_device,
                    device_properties.totalGlobalMem);
        return device_properties.totalGlobalMem;
}

#if NPP_NEW_API
EXTERN_C void init_npp_context(NppStreamContext *nppStreamCtx,
                               cudaStream_t stream)
{
        memset(nppStreamCtx, 0, sizeof *nppStreamCtx);

        nppStreamCtx->hStream = stream;
        CHECK_CUDA(cudaGetDevice(&nppStreamCtx->nCudaDeviceId));
        CHECK_CUDA(cudaDeviceGetAttribute(
            &nppStreamCtx->nCudaDevAttrComputeCapabilityMajor,
            cudaDevAttrComputeCapabilityMajor, nppStreamCtx->nCudaDeviceId));
        CHECK_CUDA(cudaStreamGetFlags(nppStreamCtx->hStream,
                                      &nppStreamCtx->nStreamFlags));

        struct cudaDeviceProp oDeviceProperties;
        cudaGetDeviceProperties(&oDeviceProperties,
                                nppStreamCtx->nCudaDeviceId);
        nppStreamCtx->nMultiProcessorCount =
            oDeviceProperties.multiProcessorCount;
        nppStreamCtx->nMaxThreadsPerMultiProcessor =
            oDeviceProperties.maxThreadsPerMultiProcessor;
        nppStreamCtx->nMaxThreadsPerBlock =
            oDeviceProperties.maxThreadsPerBlock;
        nppStreamCtx->nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
}
#endif
