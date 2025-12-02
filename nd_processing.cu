#include "nd_processing.cuh"

#include <cassert>
#include <cmath>

#include "cuda_common.cuh"    // for bilinear_sample
#include "defs.h"
#include "utils.h"

struct ramp_item {
        float val;
        int col;
};

static const __constant__ struct ramp_item ramp_ndvi[] = {
        {-0.5, 0x0c0c0c},
        {-0.2, 0xbfbfbf},
        {-0.1, 0xdbdbdb},
        {0, 0xeaeaea},
        {0.025, 0xfff9cc},
        {0.05, 0xede8b5},
        {0.075, 0xddd89b},
        {0.1, 0xccc682},
        {0.125, 0xbcb76b},
        {0.15, 0xafc160},
        {0.175, 0xa3cc59},
        {0.2, 0x91bf51},
        {0.25, 0x7fb247},
        {0.3, 0x70a33f},
        {0.35, 0x609635},
        {0.4, 0x4f892d},
        {0.45, 0x3f7c23},
        {0.5, 0x306d1c},
        {0.55, 0x216011},
        {0.6, 0x0f540a},
        {1, 0x004400},
        {INFINITY, 0x004400},
};

template <enum nd_feature feature>
__device__ void process_mapping(uint8_t *out, float val);

template <> __device__ void process_mapping<ND_UNKNOWN>(uint8_t *out, float val)
{
#ifdef GAMMA
        val = pow(val, GAMMA);
#endif
        uint8_t grayscale = 255 * __saturatef((val + 1.F) / 2.F);
        out[0] = grayscale;
        out[1] = grayscale;
        out[2] = grayscale;
}

template <> __device__ void process_mapping<NDVI>(uint8_t *out, float val)
{
        int col = ramp_ndvi[0].col;
        unsigned i = 0;
        for (i = 0; i < ARR_SIZE(ramp_ndvi) - 1; ++i) {
                if (val > ramp_ndvi[i].val && val <= ramp_ndvi[i + 1].val) {
                        col = ramp_ndvi[i].col;
                        break;
                }
        }
        uint8_t r = col >> 16;
        uint8_t g = (col >> 8) & 0xff;
        uint8_t b = col & 0xff;
        out[0] = r;
        out[1] = g;
        out[2] = b;
}

template<enum nd_feature feature>
static __global__ void nd_process(struct dec_image out,
                                      struct dec_image in1,
                                      struct dec_image in2)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width|| out_y >= out.height) {
                return;
        }

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        float abs_pos_src_x = rel_pos_src_x * in1.width;
        float abs_pos_src_y = rel_pos_src_y * in1.height;
        float val1 = bilinearSample(
            in1.data, in1.width, 1, in1.height, abs_pos_src_x, abs_pos_src_y);
        abs_pos_src_x = rel_pos_src_x * in2.width;
        abs_pos_src_y = rel_pos_src_y * in2.height;
        float val2 = bilinearSample(
            in2.data, in2.width, 1, in2.height, abs_pos_src_x, abs_pos_src_y);

        val1 /= 255.0F;
        val2 /= 255.0F;
#ifdef GAMMA
        val1 = pow(val1, 1.0F / GAMMA);
        val2 = pow(val2, 1.0F / GAMMA);
#endif

        float res = (val1 - val2) / (val1 + val2 + 0.000001f);

        process_mapping<feature>(out.data + (3 * (out_x + out_y * out.width)),
                                 res);
}

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream)
{
        GPU_TIMER_START(process_nd_features_cuda, LL_DEBUG, stream);
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        if (feature == NDVI) {
                nd_process<NDVI><<<grid, block, 0, stream>>>(*out, *in1, *in2);
        } else {
                nd_process<ND_UNKNOWN><<<grid, block, 0, stream>>>(*out, *in1, *in2);
        }
        CHECK_CUDA(cudaGetLastError());
        GPU_TIMER_STOP(process_nd_features_cuda);
}
