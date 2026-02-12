#include "nd_processing.h"

#include <cassert>
#include <cmath>
#include <cstdlib>  // for getenv

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

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
    {0.025,     0xfff9cc},
    {0.075,     0xddd89b},
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

template <enum combined_feature feature>
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

template<bool is_alpha>
static __device__ auto get_sample(const struct input_band *band,
                                   float rel_pos_src_x, float rel_pos_src_y)
{
        const float abs_pos_src_x = rel_pos_src_x * band->width;
        const float abs_pos_src_y = rel_pos_src_y * band->height;
        if constexpr (is_alpha) {
                return bilinearSample<uint8_t>((uint8_t *)band->alpha,
                                               band->width, 1, band->height,
                                               abs_pos_src_x, abs_pos_src_y);
        } else {
                return bilinearSample<uint16_t, float>(
                    (uint16_t *)band->data, band->width, 1, band->height,
                    abs_pos_src_x, abs_pos_src_y);
        }
}

static __device__ void process_hons(uint8_t *out, float r, float g, float b) {
        out[0] = __saturatef(cbrtf(0.6F * r - 0.035F)) * 255;
        out[1] = __saturatef(cbrtf(0.6F * g - 0.035F)) * 255;
        out[2] = __saturatef(cbrtf(0.6F * b - 0.035F)) * 255;
}

static __device__ void process_ndsi(uint8_t *out, float val, float g,
                                    const struct conbimend_data *d,
                                    float rel_pos_src_x, float rel_pos_src_y)
{
        if (val > 0.42) {
                out[0] = 0;
                out[1] = 0.8f * 255;
                out[2] = 255;
                return;
        }
        enum {
                R_B04_IDX = 2,
                B_B02_IDX = 3,
        };
        float r = get_sample<false>(&d->img[R_B04_IDX], rel_pos_src_x,
                                    rel_pos_src_y);
        float b = get_sample<false>(&d->img[B_B02_IDX], rel_pos_src_x,
                                    rel_pos_src_y);
        r /= 65535.0f;
        b /= 65535.0f;

        out[0] = __saturatef(r * 2.5f) * 255;
        out[1] = __saturatef(g * 2.5f) * 255;
        out[2] = __saturatef(b * 2.5f) * 255;
}

template <enum combined_feature feature>
static __global__ void nd_process(struct dec_image out, struct conbimend_data d,
                                  int d_fill_color)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width || out_y >= out.height) {
                return;
        }
        uint8_t *out_p = out.data + ((size_t)3 * (out_x + (out_y * out.width)));

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        uint8_t alpha1 = get_sample<true>(&d.img[0], rel_pos_src_x,
                                          rel_pos_src_y);
        out.alpha[out_x + (out_y * out.width)] = alpha1;
        if (alpha1 != 255) {
                out_p[0] = out_p[1] = out_p[2] = d_fill_color;
                return;
        }

        float val1 = get_sample<false>(&d.img[0], rel_pos_src_x, rel_pos_src_y);
        float val2 = get_sample<false>(&d.img[1], rel_pos_src_x, rel_pos_src_y);

        float res = (val1 - val2) / (val1 + val2 + 0.000000001f);

        if constexpr (feature == HONS) {
                float r = val1 / 65535.0f;
                float g = val2 / 65535.0f;
                float b = get_sample<false>(&d.img[2], rel_pos_src_x,
                                            rel_pos_src_y) /
                          65535.0f;
                process_hons(out_p, r, g, b);
        } else if constexpr (feature == NDSI) {
                float g = val1 / 65535.0f;
                process_ndsi(out_p, res, g, &d, rel_pos_src_x, rel_pos_src_y);
        } else {
                process_mapping<feature>(out_p, res);
        }
}

const static __constant__ uint32_t classify_map[] = {
    0x000000, // 0: No Data (Missing data)
    0xff0000, // 1: Saturated or defective pixel
    0x2f2f2f, // 2: Topographic casted shadows (called "Dark features/Shadows" for
              // data before 2022-01-25)
    0x643200, // 3: Cloud shadows
    0x00a000, // 4: Vegetation
    0xffe65a, // 5: Not-vegetated
    0x0000ff, // 6: Water
    0x808080, // 7: Unclassified
    0xc0c0c0, // 8: Cloud medium probability
    0xffffff, // 9: Cloud high probability
    0x64c8ff, // 10: Thin cirrus
    0xff96ff, // 11: Snow or ice
};

struct classify {
        const uint8_t *in_ptr;
        const uint8_t *in_alpha_ptr;
        uint8_t *out_rgb_ptr;
        uint8_t *out_alpha_ptr;
        uint8_t fill_color;

        classify(const uint8_t *in, const uint8_t *alpha, uint8_t *rgb,
                 uint8_t *oa, uint8_t fc)
            : in_ptr(in), in_alpha_ptr(alpha), out_rgb_ptr(rgb),
              out_alpha_ptr(oa), fill_color(fc)
        {
        }

        __device__ void operator()(size_t pixel_index) const
        {
                size_t rgb_idx = pixel_index * 3;

                uint32_t val = 0;
                unsigned index = in_ptr[pixel_index];
                if (index < countof(classify_map)) {
                        val = classify_map[index];
                }
                /// @note on missing data (index == 0), we set alpha=0 + fill
                /// color; in custom scripts they set [0,0,0], also for index>11
                /// (out-of-bound)
                if (val == 0 || in_alpha_ptr[pixel_index] != 255) {
                        out_rgb_ptr[rgb_idx + 0] = fill_color; // R
                        out_rgb_ptr[rgb_idx + 1] = fill_color; // G
                        out_rgb_ptr[rgb_idx + 2] = fill_color; // B
                        out_alpha_ptr[pixel_index] = 0;
                        return;
                }

                out_alpha_ptr[pixel_index] = 255;
                out_rgb_ptr[rgb_idx + 0] = val >> 16;
                out_rgb_ptr[rgb_idx + 1] = (val >> 8) & 0xFF;
                out_rgb_ptr[rgb_idx + 2] = val & 0xFF;
        }
};

static void process_scl(struct dec_image *out, const struct input_band *in,
                        cudaStream_t stream)
{
        GPU_TIMER_START(process_nd_features_cuda_scl, LL_DEBUG, stream);

        size_t count = (size_t)in->width * in->height;
        thrust::for_each(thrust::cuda::par.on(stream),
                         thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(count),
                         classify{(uint8_t *)in->data, in->alpha, out->data,
                                  out->alpha, fill_color});

        GPU_TIMER_STOP(process_nd_features_cuda_scl);
}

void process_nd_features_cuda(struct dec_image *out, enum combined_feature feature,
                              const struct conbimend_data *in, cudaStream_t stream)
{
        if (feature == SCL) {
                assert(in->count == 1);
                assert(!in->is_16b);
                process_scl(out, &in->img[0], stream);
                return;
        }

        assert(in->is_16b);
        assert(feature != HONS || in->count == 3); //  HONS -> count=3
        assert(feature != NDSI || in->count == 4); //  NDSI -> count=4
        assert((feature == HONS || feature == NDSI) || in->count == 2); // !NDSI and !HONS -> count=2
        assert(in->img[0].alpha != nullptr);
        GPU_TIMER_START(process_nd_features_cuda, LL_DEBUG, stream);
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        auto *fn = nd_process<ND_UNSPEC>;
        switch (feature) {
        case FEAT_NONE:
        case SCL: // already handled
                abort();
        case HONS:
                fn =  nd_process<HONS>;
                break;
        case NDVI:
                fn =  nd_process<NDVI>;
                break;
        case NDMI:
                fn = nd_process<NDMI>;
                break;
        case NDWI:
                fn = nd_process<NDWI>;
                break;
        case NDSI:
                fn = nd_process<NDSI>;
                break;
        case ND_UNSPEC: // already set
                break;
        }
        if (getenv("GRAYSCALE") != nullptr) {
                fn = nd_process<ND_UNSPEC>;
        }
        fn<<<grid, block, 0, stream>>>(*out, *in, fill_color);
        CHECK_CUDA(cudaGetLastError());
        GPU_TIMER_STOP(process_nd_features_cuda);
}
