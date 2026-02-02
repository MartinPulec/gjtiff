#include "rotate_utm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#ifndef NDEBUG
#define NDEBUG
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>
#undef NDEBUG
#else
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>
#endif
#include <errno.h>
#include <limits.h> // PATH_MAX
#include <npp.h>
#include <ogr_srs_api.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "cuda_common.cuh"
#include "defs.h"
#include "nppdefs.h"
#include "nppi_geometry_transforms.h"
#include "rotate.h" // for get_lat_lon_min_max
#include "utils.h"

using namespace std;

extern long long mem_limit; // defined in main.c

struct rotate_utm_state {
        cudaStream_t stream;
        NppStreamContext nppStreamCtx;
        OGRSpatialReferenceH src_srs;
        OGRSpatialReferenceH dst_srs;
};

struct rotate_utm_state *rotate_utm_init(cudaStream_t stream)
{
        struct rotate_utm_state *s = (struct rotate_utm_state *)calloc(
            1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

        init_npp_context(&s->nppStreamCtx, stream);

        s->src_srs = OSRNewSpatialReference(nullptr);
        s->dst_srs = OSRNewSpatialReference(nullptr);
        OSRImportFromEPSG(s->dst_srs, EPSG_WEB_MERCATOR);

        return s;
}

void rotate_utm_destroy(struct rotate_utm_state *s)
{
        if (s == NULL) {
                return;
        }
        OSRDestroySpatialReference(s->src_srs);
        OSRDestroySpatialReference(s->dst_srs);
        free(s);
}

struct bounds {
        float bound[4];
};
using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;

template <typename T, int components, bool alpha>
static __global__ void
kernel_utm_to_web_mercator(device_projection const d_proj, const void *d_in_v,
                           void *d_out_v, uint8_t *d_out_alpha, int in_width,
                           int in_height, int out_width, int out_height,
                           struct bounds src_bounds, struct bounds dst_bounds,
                           int fill_color)
{
        const auto *d_in = (T const *)d_in_v;
        auto *d_out = (T *)d_out_v;
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out_width || out_y >= out_height) {
                return;
        }

        float y_scale = dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP];
        float this_y = dst_bounds.bound[YTOP];
        this_y += y_scale * ((out_y + .5f) / out_height);

        float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
        float this_x = dst_bounds.bound[XLEFT];
        this_x += x_scale * ((out_x + .5f) / out_width);

        // transformace
        float pos_wgs84_lon = 360. * this_x - 180.; // lambda
        float t = (float) M_PI * (1. - 2. *  this_y);
        float fi_rad = 2 * atanf(powf(M_E, t)) - (M_PI / 2);
        float pos_wgs84_lat = fi_rad * 180. / M_PI;

        cuproj::vec_2d<float> pos_wgs84{pos_wgs84_lat, pos_wgs84_lon};
        cuproj::vec_2d<float> pos_utm = d_proj.transform(pos_wgs84);
        pos_utm = d_proj.transform(pos_wgs84);

        float rel_pos_src_x = (pos_utm.x - src_bounds.bound[XLEFT]) /
                              (src_bounds.bound[XRIGHT] - src_bounds.bound[XLEFT]);
        float rel_pos_src_y = (pos_utm.y - src_bounds.bound[YTOP]) /
                              (src_bounds.bound[YBOTTOM] - src_bounds.bound[YTOP]);

        if (rel_pos_src_x < 0 || rel_pos_src_x > 1 ||
            rel_pos_src_y < 0 || rel_pos_src_y > 1) {
                for (int i = 0; i < components; ++i) {
                        d_out[(components * (out_x + out_y * out_width)) +
                              i] = fill_color << ((sizeof(T) - 1) * CHAR_BIT);
                }
                if (alpha) {
                        d_out_alpha[out_x + (out_y * out_width)] = 0;
                }
                return;
        }
        if (alpha) {
                d_out_alpha[out_x + (out_y * out_width)] = 255;
        }
        // if (out_y == 0) {
        //         printf("%f %f\n" , rel_pos_src_x, rel_pos_src_y);
        // }

        float abs_pos_src_x = rel_pos_src_x * in_width;
        float abs_pos_src_y = rel_pos_src_y * in_height;

        for (int i = 0; i < components; ++i) {
                d_out[(components * (out_x + out_y * out_width)) + i] =
                    bilinearSample(d_in + i, in_width, components, in_height,
                                   abs_pos_src_x, abs_pos_src_y);
        }
}

bool transform_to_float(double *x, double *y,
                        OGRCoordinateTransformationH transform)
{
        constexpr double M = EARTH_PERIMETER;
        if (OCTTransform(transform, 1, x, y, NULL)) {
                *x = (*x + M) / (2 * M);
                // flip from bottom-up
                *y = 1 - ((*y + M) / (2 * M));
                return true;
        }
        ERROR_MSG("Cannot tranform!\n");
        return false;
}

/// @todo needed? isn't sufficient to compute just left/top and right/bottom corner
static bool adjust_dst_bounds(int x_loc, int y_loc,
                              const struct bounds *src_bounds,
                              struct bounds *dst_bounds,
                              OGRCoordinateTransformationH transform)
{
        double x = src_bounds->bound[x_loc];
        double y = src_bounds->bound[y_loc];
        if (!transform_to_float(&x, &y, transform)) {
                return false;
        }
        auto xf = (float)x;
        auto yf = (float)y;
        dst_bounds->bound[XLEFT] = min(dst_bounds->bound[XLEFT], xf);
        dst_bounds->bound[XRIGHT] = max(dst_bounds->bound[XRIGHT], xf);
        dst_bounds->bound[YTOP] = min(dst_bounds->bound[YTOP], yf);
        dst_bounds->bound[YBOTTOM] = max(dst_bounds->bound[YBOTTOM], yf);
        return true;
}

struct owned_image *rotate_utm(struct rotate_utm_state *s,
                               const struct dec_image *in)
{
        GPU_TIMER_START(utm_to_epsg_3857, LL_DEBUG, s->stream);

        if (in->comp_count != 1 && in->comp_count != 3) {
                ERROR_MSG("unsupporeted component count %d!", in->comp_count);
                return nullptr;
        }
        const double src_ratio = (double) in->width / in->height;

        struct bounds src_bounds{};
        src_bounds.bound[XLEFT] = in->bounds[XLEFT];
        src_bounds.bound[YTOP] = in->bounds[YTOP];
        src_bounds.bound[XRIGHT] = in->bounds[XRIGHT];
        src_bounds.bound[YBOTTOM] = in->bounds[YBOTTOM];

        struct bounds dst_bounds{};
        dst_bounds.bound[XLEFT] = 1;
        dst_bounds.bound[YTOP] = 1;
        dst_bounds.bound[XRIGHT] = 0;
        dst_bounds.bound[YBOTTOM] = 0;
        OSRImportFromEPSG(s->src_srs, atoi(strchr(in->authority, ':') + 1));
        OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(
            s->src_srs, s->dst_srs);
        if (transform == nullptr) {
                ERROR_MSG("[roate_utm] Cannot create transform!\n");
                abort();
        }
        bool succeed = adjust_dst_bounds(XLEFT, YTOP, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XRIGHT, YTOP, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XRIGHT, YBOTTOM, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XLEFT, YBOTTOM, &src_bounds, &dst_bounds, transform);
        if (!succeed) {
                ERROR_MSG("[rotate_utm] Failed to adjust dst bounds!\n");
                abort();
        }
        double dst_ratio = (dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT]) /
                           (dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP]);

        using coordinate = cuproj::vec_2d<float>;
        auto proj = cuproj::make_projection<coordinate>( "EPSG:4326", in->authority );
        auto d_proj = proj->get_device_projection(cuproj::direction::FORWARD);

        struct dec_image dst_desc = *in;
        if (in->e3857_sug_w != 0 && in->e3857_sug_h != 0) {
                dst_desc.width = in->e3857_sug_w;
                dst_desc.height = in->e3857_sug_h;
        } else {
                dst_desc.width = in->width;
                dst_desc.height = in->height;
                if (dst_ratio >= src_ratio) {
                        dst_desc.width *= dst_ratio;
                } else {
                        dst_desc.height /= dst_ratio;
                }
        }

        dst_desc.alpha = alpha_wanted ? (unsigned char *) 1 : nullptr;
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        snprintf(ret->img.authority, sizeof ret->img.authority, "%s", "EPSG:3857");
        for (unsigned i = 0; i < ARR_SIZE(dst_bounds.bound); ++i) {
                ret->img.bounds[i] = dst_bounds.bound[i];
        }

        dim3 block(16, 16);
        int width = dst_desc.width;
        int height = dst_desc.height;
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        decltype(kernel_utm_to_web_mercator<uint8_t, 1, true>) *kernel = nullptr;
#define GET_KERNEL(comp_count, alpha)                                          \
        (in->is_16b ? kernel_utm_to_web_mercator<uint16_t, comp_count, alpha>  \
                    : kernel_utm_to_web_mercator<uint8_t, comp_count, alpha>)
        if (alpha_wanted) {
                if (in->comp_count == 1) {
                        kernel = GET_KERNEL(1, true);
                } else {
                        kernel = GET_KERNEL(3, true);
                }
        } else {
                if (in->comp_count == 1) {
                        kernel = GET_KERNEL(1, false);
                } else {
                        kernel = GET_KERNEL(3, false);
                }
        }
        kernel<<<grid, block, 0, s->stream>>>(
            d_proj, in->data, ret->img.data, ret->img.alpha, in->width,
            in->height, width, height, src_bounds, dst_bounds, fill_color);
        CHECK_CUDA(cudaGetLastError());

        // Cleanup
        OCTDestroyCoordinateTransformation(transform);

        GPU_TIMER_STOP(utm_to_epsg_3857);

        delete proj;

        return ret;
}

