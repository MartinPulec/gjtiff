#include "nd_processing.cuh"

#include <cassert>
#include <cmath>
#include <cstdlib>  // for getenv

#include "cuda_common.cuh"    // for bilinear_sample
#include "defs.h"
#include "utils.h"

struct ramp_item {
        float val;
        int col;
};

static const __constant__ struct ramp_item ramp_ndvi[] = {
    {-INFINITY, 0x0c0c0c},
    {-1,        0x0c0c0c},
    {-0.5,      0xbfbfbf},
    {-0.2,      0xdbdbdb},
    {0,         0xeaeaea},
    {0.1,       0xccc682},
    {0.2,       0x91bf51},
    {1,         0x004400},
    {INFINITY,  0x004400},
};

static const __constant__ struct ramp_item ramp_ndmi[] = {
    {-INFINITY, 0x800000},
    {-0.8,      0x800000},
    {-0.24,     0xff0000},
    {-0.032,    0xffff00},
    {0.032,     0x00ffff},
    {0.24,      0x0000ff},
    {0.8,       0x000080},
    {INFINITY,  0x000080},
};

static const __constant__ struct ramp_item ramp_ndwi[] = {
    {-INFINITY, 0x008000},
    {-0.8,      0x008000},
    {0,         0xFFFFFF},
    {0.8,       0x0000CC},
    {INFINITY,  0x0000CC},
};

template <enum nd_feature feature>
__device__ void process_mapping(uint8_t *out, float val);

template <> __device__ void process_mapping<ND_UNSPEC>(uint8_t *out, float val)
{
        val = __saturatef((val + 1.F) / 2.F);
#ifdef GAMMA
        val = pow(val, GAMMA);
#endif
        uint8_t grayscale = 255 * val;
        out[0] = grayscale;
        out[1] = grayscale;
        out[2] = grayscale;
}

// __device__ static void process_mapping_nd(uint8_t *out, float val,
//                                           const struct ramp_item *ramp)
// {
//         int col = ramp[0].col;
//         while (ramp->val != INFINITY) {
//                 if (val > ramp->val && val <= ramp[1].val) {
//                         col = ramp[1].col;
//                         break;
//                 }
//                 ramp++;
//         }
//         uint8_t r = col >> 16;
//         uint8_t g = (col >> 8) & 0xff;
//         uint8_t b = col & 0xff;
// #ifdef GAMMA
//         // r = 255.0 * pow(r / 255.0, GAMMA);
//         // g = 255.0 * pow(g / 255.0, GAMMA);
//         // b = 255.0 * pow(b / 255.0, GAMMA);
// #endif
//         out[0] = r;
//         out[1] = g;
//         out[2] = b;
// }

__device__ static void
process_mapping_nd_interpolate(uint8_t *out, float val,
                               const struct ramp_item *ramp)
{
        // val -= 0.1;
        while (ramp->val != INFINITY) {
                if (val > ramp->val && val <= ramp[1].val) {
                        break;
                }
                ramp++;
        }
        const struct ramp_item *ramp1 = ramp;
        const struct ramp_item *ramp2 = ramp + 1;
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        if (isinf(ramp1->val) || isinf(ramp2->val)) {
                const struct ramp_item *cur_ramp = isinf(ramp1->val) ? ramp1 : ramp2;
                int col = cur_ramp->col;
                r = col >> 16;
                g = (col >> 8) & 0xff;
                b = col & 0xff;
        } else {
                int col1 = ramp1->col;
                uint8_t r1 = col1 >> 16;
                uint8_t g1 = (col1 >> 8) & 0xff;
                uint8_t b1 = col1 & 0xff;
                int col2 = ramp2->col;
                uint8_t r2 = col2 >> 16;
                uint8_t g2 = (col2 >> 8) & 0xff;
                uint8_t b2 = col2 & 0xff;
                float scale =  ramp2->val - ramp1->val;
                val -= ramp1->val;
                r = round(r1 * (scale - val) / scale + r2 * val / scale);
                g = round(g1 * (scale - val) / scale + g2 * val / scale);
                b = round(b1 * (scale - val) / scale + b2 * val / scale);
                // if (val > 0.4) printf("%d %d = %d  %d %d   %f %f   %f\n", b1, b2, r,g,b,  ramp[0].val, ramp[1].val, val);
        }

        // int col = r << 16 | g << 8 | b;

        // uint8_t r = col >> 16;
        // uint8_t g = (col >> 8) & 0xff;
        // uint8_t b = col & 0xff;
#ifdef GAMMA
        // r = 255.0 * pow(r / 255.0, GAMMA);
        // g = 255.0 * pow(g / 255.0, GAMMA);
        // b = 255.0 * pow(b / 255.0, GAMMA);
#endif
        out[0] = r;
        out[1] = g;
        out[2] = b;
}
template <> __device__ void process_mapping<NDVI>(uint8_t *out, float val)
{
        process_mapping_nd_interpolate(out, val, ramp_ndvi);
}

template <> __device__ void process_mapping<NDMI>(uint8_t *out, float val)
{
        process_mapping_nd_interpolate(out, val, ramp_ndmi);
}


template <> __device__ void process_mapping<NDWI>(uint8_t *out, float val)
{
        process_mapping_nd_interpolate(out, val, ramp_ndwi);
}

template <enum nd_feature feature>
static __global__ void nd_process(struct dec_image out, struct dec_image in1,
                                  struct dec_image in2, int d_fill_color)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width || out_y >= out.height) {
                return;
        }
        uint8_t *out_p = out.data + ((size_t)3 * (out_x + (out_y * out.width)));

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        float abs_pos_src_x = rel_pos_src_x * in1.width;
        float abs_pos_src_y = rel_pos_src_y * in1.height;
        uint8_t alpha1 = bilinearSample(in1.alpha, in1.width, 1, in1.height,
                                        abs_pos_src_x, abs_pos_src_y);
        out.alpha[out_x + (out_y * out.width)] =  alpha1;
        if (alpha1 != 255) {
                out_p[0] = out_p[1] = out_p[2] = d_fill_color;
                return;
        }

        float val1 = bilinearSample<uint16_t, float>(
            (uint16_t *)in1.data, in1.width, 1, in1.height, abs_pos_src_x,
            abs_pos_src_y);

        abs_pos_src_x = rel_pos_src_x * in2.width;
        abs_pos_src_y = rel_pos_src_y * in2.height;
        float val2 = bilinearSample<uint16_t, float>(
            (uint16_t *)in2.data, in2.width, 1, in2.height, abs_pos_src_x,
            abs_pos_src_y);

        float res = (val1 - val2) / (val1 + val2 + 0.000000001f);

        process_mapping<feature>(out_p, res);
}

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream)
{
        assert(alpha_wanted);
        assert(in1->alpha != nullptr);
        assert(in1->is_16b && in2->is_16b);
        GPU_TIMER_START(process_nd_features_cuda, LL_DEBUG, stream);
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        auto *fn = nd_process<ND_UNSPEC>;
        switch (feature) {
        case ND_NONE:
                abort();
        case NDVI:
                fn =  nd_process<NDVI>;
                break;
        case NDMI:
                fn = nd_process<NDMI>;
                break;
        case NDWI:
                fn = nd_process<NDWI>;
                break;
        case ND_UNSPEC: // already set
                break;
        }
        if (getenv("GRAYSCALE") != nullptr) {
                fn = nd_process<ND_UNSPEC>;
        }
        fn<<<grid, block, 0, stream>>>(*out, *in1, *in2, fill_color);
        CHECK_CUDA(cudaGetLastError());
        GPU_TIMER_STOP(process_nd_features_cuda);
}
