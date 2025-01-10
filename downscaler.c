#include "downscaler.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include "kernels.h"
#include "utils.h"

struct downscaler_state {
        cudaStream_t stream;
        int factor;

        uint8_t *output;
        size_t output_allocated;
};

struct downscaler_state *downscaler_init(int downscale_factor,
                                         cudaStream_t stream)
{
        struct downscaler_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->factor = downscale_factor;
        s->stream = stream;

        return s;
}

struct dec_image downscale(struct downscaler_state *s,
                           const struct dec_image *in)
{
        struct dec_image downscaled = { 0 };
        downscaled.comp_count = in->comp_count;
        downscaled.width = in->width / s->factor;
        downscaled.height = in->height / s->factor;
        size_t required_size = (size_t)downscaled.comp_count *
                               downscaled.width * downscaled.height;
        if (required_size > s->output_allocated) {
                CHECK_CUDA(cudaFreeAsync(s->output, s->stream));
                CHECK_CUDA(cudaMallocAsync((void **)&s->output, required_size,
                                           s->stream));
                s->output_allocated = required_size;
        }
        downscaled.data = s->output;
        downscale_image_cuda(in->data, downscaled.data, in->comp_count,
                             in->width, in->height, s->factor, s->stream);
        return downscaled;
}

void downscaler_destroy(struct downscaler_state *s) {
        if (s == NULL) {
                return;
        }
        cudaFreeAsync(s->output, s->stream);
        free(s);
}
