#include "downscaler.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "defs.h"
#include "kernels.h"
#include "utils.h"

extern int interpolation; // defined in main.c

struct downscaler_state {
        cudaStream_t stream;

        uint8_t *output;
        size_t output_allocated;
};

struct downscaler_state *downscaler_init(cudaStream_t stream)
{
        struct downscaler_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

        return s;
}

struct dec_image downscale(struct downscaler_state *s, int downscale_factor,
                           const struct dec_image *in)
{
        struct dec_image downscaled = *in;
        downscaled.comp_count = in->comp_count;
        downscaled.width = in->width / downscale_factor;
        downscaled.height = in->height / downscale_factor;
        size_t required_size = (size_t)downscaled.comp_count *
                               downscaled.width * downscaled.height;
        if (required_size > s->output_allocated) {
                CHECK_CUDA(cudaFreeAsync(s->output, s->stream));
                CHECK_CUDA(cudaMallocAsync((void **)&s->output, required_size,
                                           s->stream));
                s->output_allocated = required_size;
        }
        downscaled.data = s->output;
        GPU_TIMER_START(downscale, LL_DEBUG, s->stream);
#ifndef DOWNSCALE_NO_NPP
        if (nppGetStream() != s->stream) {
                nppSetStream(s->stream);
        }
        NppiSize srcSize = {in->width, in->height};
        NppiRect srcROI = {0, 0, in->width, in->height};
#if NPP_VERSION_MAJOR <= 8
        double factor = 1.0 / downscale_factor;
#else
        NppiRect dstROI = {0, 0, downscaled.width, downscaled.height};
#endif
        NppiSize dstSize = {downscaled.width, downscaled.height};
        size_t srcPitch = (size_t)in->width * in->comp_count;
        size_t dstPitch = (size_t)downscaled.width * in->comp_count;

        assert(in->comp_count == 1 || in->comp_count == 3);
        const NppiInterpolationMode imode = interpolation > 0
                                                ? interpolation
                                                : NPPI_INTER_SUPER;
        if (in->comp_count == 1) {
#if NPP_VERSION_MAJOR <= 8
                CHECK_NPP(nppiResize_8u_C1R(in->data, srcSize, srcPitch, srcROI,
                                            downscaled.data, dstPitch, dstSize,
                                            factor, factor, imode));
#else
                CHECK_NPP(nppiResize_8u_C1R(in->data, srcPitch, srcSize, srcROI,
                                            downscaled.data, dstPitch, dstSize,
                                            dstROI, imode));
#endif
        } else {
#if NPP_VERSION_MAJOR <= 8
                CHECK_NPP(nppiResize_8u_C3R(in->data, srcSize, srcPitch, srcROI,
                                            downscaled.data, dstPitch, dstSize,
                                            factor, factor, imode));
#else
                CHECK_NPP(nppiResize_8u_C3R(in->data, srcPitch, srcSize, srcROI,
                                            downscaled.data, dstPitch, dstSize,
                                            dstROI, imode));
#endif
        }
#else
        downscale_image_cuda(in->data, downscaled.data, in->comp_count,
                             in->width, in->height, downscale_factor,
                             s->stream);
#endif
        GPU_TIMER_STOP(downscale);
        return downscaled;
}

void downscaler_destroy(struct downscaler_state *s) {
        if (s == NULL) {
                return;
        }
        cudaFreeAsync(s->output, s->stream);
        free(s);
}
