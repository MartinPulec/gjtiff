#include "downscaler.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <stdint.h>
#include <stdlib.h>

#include "defs.h"
#include "kernels.h"
#include "utils.h"

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
        struct dec_image downscaled = { 0 };
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
        TIMER_START(downscale, LL_DEBUG);
#ifndef DOWNSCALE_NO_NPP
        if (nppGetStream() != s->stream) {
                nppSetStream(s->stream);
        }
        NppiSize srcSize = {in->width, in->height};
        NppiRect srcROI = {0, 0, in->width, in->height};
        NppiRect dstROI = {0, 0, downscaled.width, downscaled.height};
        NppiSize dstSize = {downscaled.width, downscaled.height};

        double factor = 1.0 / downscale_factor;

#if NPP_VERSION_MAJOR <= 8
        CHECK_NPP(nppiResize_8u_C1R(in->data, srcSize, in->width, srcROI,
                                    downscaled.data, downscaled.width, dstSize,
                                    factor, factor, NPPI_INTER_SUPER));
#else
        CHECK_NPP(nppiResize_8u_C1R(in->data, in->width, srcSize, srcROI,
                                    downscaled.data, downscaled.width, dstSize,
                                    dstROI, NPPI_INTER_SUPER));
#endif
#else
        downscale_image_cuda(in->data, downscaled.data, in->comp_count,
                             in->width, in->height, downscale_factor,
                             s->stream);
#endif
        TIMER_STOP(downscale);
        return downscaled;
}

void downscaler_destroy(struct downscaler_state *s) {
        if (s == NULL) {
                return;
        }
        cudaFreeAsync(s->output, s->stream);
        free(s);
}
