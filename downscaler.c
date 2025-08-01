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

#ifdef NPP_NEW_API
#define CONTEXT , s->nppStreamCtx
#else
#define CONTEXT
#endif

extern int interpolation; // defined in main.c

struct downscaler_state {
        cudaStream_t stream;
#ifdef NPP_NEW_API
        NppStreamContext nppStreamCtx;
#endif

        uint8_t *output;
        size_t output_allocated;
};

struct downscaler_state *downscaler_init(cudaStream_t stream)
{
        struct downscaler_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;
#ifdef NPP_NEW_API
        init_npp_context(&s->nppStreamCtx, stream);
#endif

        return s;
}

static void downscale_int(struct downscaler_state *s, int new_width,
                          size_t dstPitch, int new_height,
                          const struct dec_image *in, uint8_t *d_output)
{
        GPU_TIMER_START(downscale, LL_DEBUG, s->stream);
#ifndef DOWNSCALE_NO_NPP
#ifndef NPP_NEW_API
        if (nppGetStream() != s->stream) {
                nppSetStream(s->stream);
        }
#endif
        NppiSize srcSize = {in->width, in->height};
        NppiRect srcROI = {0, 0, in->width, in->height};
#if NPP_VERSION_MAJOR <= 8
#error "no longer implemented"
        double factor = 1.0 / downscale_factor;
#else
        NppiRect dstROI = {0, 0, new_width, new_height};
#endif
        NppiSize dstSize = {new_width, new_height};
        size_t srcPitch = (size_t)in->width * in->comp_count;

        assert(in->comp_count == 1 || in->comp_count == 3);
        // was NPPI_INTER_SUPER but doesnt seem to work for all resolutions,
        // namely upscaling doesn't (for some factor?) work. If used,
        // NPP_RESIZE_FACTOR_ERROR should be perhaps handled with bilinear
        // (or other) fallback
        const NppiInterpolationMode imode = interpolation > 0
                                                ? interpolation
                                                : NPPI_INTER_LINEAR;

        NppStatus (*npp_resize)(const Npp8u *pSrc, int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) = NULL;
        if (in->comp_count == 1) {
                npp_resize = nppiResize_8u_C1R_Ctx;
        } else {
                npp_resize = nppiResize_8u_C3R_Ctx;
        }
        CHECK_NPP(npp_resize(in->data, srcPitch, srcSize, srcROI, d_output,
                             dstPitch, dstSize, dstROI, imode,
                             s->nppStreamCtx));
#else
#error "no longer implemented"
        downscale_image_cuda(in->data, downscaled.data, in->comp_count,
                             in->width, in->height, downscale_factor,
                             s->stream);
#endif
        GPU_TIMER_STOP(downscale);
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
        const size_t dstPitch = (size_t)downscaled.width * in->comp_count;
        downscale_int(s, downscaled.width, dstPitch, downscaled.height, in,
                      s->output);
        return downscaled;
}

/**
 * @param new_width new ROI width
 * @param xpitch    destination pitch in bytes
 */
struct owned_image *scale_pitch(struct downscaler_state *state, int new_width,
                                int x, size_t xpitch, int new_height,
                                int y, size_t dst_lines,
                                const struct owned_image *old)
{
        struct dec_image new_desc = old->img;
        new_desc.width = (int)xpitch / old->img.comp_count;
        new_desc.height = (int)dst_lines;
        struct owned_image *ret = new_cuda_owned_image(&new_desc);
        unsigned char *data = ret->img.data + ((ptrdiff_t)y * xpitch) +
                              ((ptrdiff_t)x * old->img.comp_count);
        downscale_int(state, new_width, xpitch, new_height, &old->img,
                      data);
        return ret;
}

struct owned_image *scale(struct downscaler_state *state, int new_width,
                          int new_height, const struct owned_image *old)
{
        const size_t dstPitch = (size_t)new_width * old->img.comp_count;
        return scale_pitch(state, new_width, 0, dstPitch, new_height, 0,
                           new_height, old);
}

void downscaler_destroy(struct downscaler_state *s) {
        if (s == NULL) {
                return;
        }
        CHECK_CUDA(cudaFreeAsync(s->output, s->stream));
        free(s);
}
