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
};

struct rotate_utm_state *rotate_utm_init(cudaStream_t stream)
{
        struct rotate_utm_state *s = (struct rotate_utm_state *)calloc(
            1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

        init_npp_context(&s->nppStreamCtx, stream);

        return s;
}

void rotate_utm_destroy(struct rotate_utm_state *s)
{
        if (s == NULL) {
                return;
        }
        free(s);
}

struct bounds {
        float bound[4];
};
using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;

#ifdef TWO_STEP_CONV
static __global__ void kernel_to_wgs84(device_projection const d_proj,
                              const uint8_t *d_in, uint8_t *d_out, int in_width,
                              int in_height, int out_width, int out_height,
                              struct bounds src_bounds,
                              struct bounds dst_bounds)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out_width || out_y >= out_height) {
                return;
        }

        float lat_scale = dst_bounds.bound[YTOP] - dst_bounds.bound[YBOTTOM];
        float this_lat = dst_bounds.bound[YTOP];
        this_lat -= lat_scale * ((out_y + .5f) / out_height);

        float lon_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
        float this_lon = dst_bounds.bound[XLEFT];
        this_lon += lon_scale * ((out_x + .5f) / out_width);

        cuproj::vec_2d<float> pos_wgs84{this_lat, this_lon};
        cuproj::vec_2d<float> pos_utm = d_proj.transform(pos_wgs84);
        pos_utm = d_proj.transform(pos_wgs84);

        float rel_pos_src_x = (pos_utm.x - src_bounds.bound[XLEFT]) /
                              (src_bounds.bound[XRIGHT] - src_bounds.bound[XLEFT]);
        float rel_pos_src_y = (pos_utm.y - src_bounds.bound[YBOTTOM]) /
                              (src_bounds.bound[YTOP] - src_bounds.bound[YBOTTOM]);
        rel_pos_src_y = 1 - rel_pos_src_y;

        // if (x  == 0 && y == 0) {
                // printf("%f %f\n\n", in.x, in.y);
                // printf("%f %f\n\n", out.x, out.y);
                // printf("%f %f\n\n", rel_pos_src_x, rel_pos_src_y);
                // printf("%f \n\n", this_lon);
        // }
        if (rel_pos_src_x < 0 || rel_pos_src_x > 1 ||
            rel_pos_src_y < 0 || rel_pos_src_y > 1) {
                d_out[out_x + out_y * out_width] = 0;
                return;
        }
        // if (out_y == 0) {
        //         printf("%f\n", dst_bounds.bound[YBOTTOM]);
        //         printf("HERE! %d %f %f %f %f\n" , out_x, this_lat, this_lon, pos_utm.x, pos_utm.y);
        //         printf("%f %f\n" , rel_pos_src_x, rel_pos_src_y);
        // }

        // if (out_x == out_width / 2 && out_y == out_height / 2) {
                // printf("%f %f\n\n", pos_wgs84.x, pos_wgs84.y);
                // printf("%f %f\n\n", pos_utm.x, pos_utm.y);
                // printf("%f %f\n\n", rel_pos_src_x, rel_pos_src_y);
        // }

        float abs_pos_src_x = rel_pos_src_x * in_width;
        float abs_pos_src_y = rel_pos_src_y * in_height;

        // if (out_x == out_width / 2 && out_y == out_height / 2) {
        //         printf("%f %f\n\n", pos_wgs84.x, pos_wgs84.y);
        //         printf("%f %f\n\n", pos_utm.x, pos_utm.y);
        //         printf("%f %f\n\n", rel_pos_src_x, rel_pos_src_y);
        //         printf("%f %f\n", abs_pos_src_x, abs_pos_src_y);
        // }

        // d_out[out_x + out_y * out_width] = d_in[out_x + out_y * in_width];
        d_out[out_x + out_y * out_width] = bilinearSample(d_in, in_width, in_height, abs_pos_src_x, abs_pos_src_y);
        // d_out[out_x + out_y * out_width] = d_in[(int) abs_pos_src_x + out_y *(int) abs_pos_src_y];
        // d_out[out_x + out_y * out_width] = rel_pos_src_y * 255;;
}

static struct owned_image *to_epsg_4326(struct rotate_utm_state *s,
                                        const struct dec_image *in)
{
        // test(); return nullptr;
        double src_ratio = (in->bounds[XRIGHT] - in->bounds[XLEFT]) /
                           (in->bounds[YTOP] - in->bounds[YBOTTOM]);
        double lat_top = max(in->coords[ULEFT].latitude, in->coords[URIGHT].latitude);
        double lat_bot = min(in->coords[BLEFT].latitude, in->coords[BRIGHT].latitude);
        double lon_left = min(in->coords[ULEFT].longitude, in->coords[BLEFT].longitude);
        double lon_right = max(in->coords[URIGHT].longitude, in->coords[BRIGHT].longitude);
        double dst_ratio = (lon_right - lon_left) / (lat_top - lat_bot);
        struct bounds dst_bounds;
        dst_bounds.bound[XLEFT] = lon_left;
        dst_bounds.bound[YTOP] = lat_top;
        dst_bounds.bound[XRIGHT] = lon_right;
        dst_bounds.bound[YBOTTOM] = lat_bot;
        struct bounds src_bounds;
        src_bounds.bound[XLEFT] = in->bounds[XLEFT];
        src_bounds.bound[YTOP] = in->bounds[YTOP];
        src_bounds.bound[XRIGHT] = in->bounds[XRIGHT];
        src_bounds.bound[YBOTTOM] = in->bounds[YBOTTOM];
        struct dec_image dst_desc = *in;
        if (dst_ratio >= src_ratio) {
                dst_desc.width = (int)(in->height * dst_ratio);
        } else {
                dst_desc.height = (int)(in->width / dst_ratio);
        }
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        for (unsigned i = 0; i < ARR_SIZE(ret->img.bounds); ++i) {
                ret->img.bounds[i] = dst_bounds.bound[i];
        }
        snprintf(ret->img.authority, sizeof ret->img.authority, "%s", "EPSG:4326");

        using coordinate = cuproj::vec_2d<float>;
        auto proj = cuproj::make_projection<coordinate>( "EPSG:4326", in->authority );
        auto d_proj = proj->get_device_projection(cuproj::direction::FORWARD);

        dim3 block(16, 16);
        int width = dst_desc.width;
        int height = dst_desc.height;
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        kernel_to_wgs84<<<grid, block, 0, s->stream>>>(d_proj,
            in->data, ret->img.data, in->width, in->height, width, height, src_bounds, dst_bounds);

        return ret;
}

static __global__ void kernel_to_web_mercator(
                              const uint8_t *d_in, uint8_t *d_out, int in_width,
                              int in_height, int out_width, int out_height,
                              struct bounds src_bounds,
                              struct bounds dst_bounds)
{
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

        float rel_pos_src_x = (pos_wgs84_lon - src_bounds.bound[XLEFT]) /
                              (src_bounds.bound[XRIGHT] -
                               src_bounds.bound[XLEFT]);
        float rel_pos_src_y = (pos_wgs84_lat - src_bounds.bound[YBOTTOM]) /
                              (src_bounds.bound[YTOP] -
                               src_bounds.bound[YBOTTOM]);

        if (rel_pos_src_x < 0 || rel_pos_src_x > 1 ||
            rel_pos_src_y < 0 || rel_pos_src_y > 1) {
                d_out[out_x + out_y * out_width] = 0;
                return;
        }

        // if (out_x == out_width / 2 && out_y == out_height / 2) {
        //         printf("%f %f\n\n", this_x, this_y);
        //         printf("%f %f\n\n", pos_wgs84_lat, pos_wgs84_lon);
        //         printf("%f %f\n\n", rel_pos_src_x, rel_pos_src_y);
        // }

        float abs_pos_src_x = rel_pos_src_x * in_width;
        rel_pos_src_y = 1 - rel_pos_src_y;
        float abs_pos_src_y = rel_pos_src_y * in_height;

        d_out[out_x + out_y * out_width] = bilinearSample(d_in, in_width, in_height, abs_pos_src_x, abs_pos_src_y);
}

// WSG84 (lon, lat) to Web Mercator
static struct owned_image *
epsg_4326_to_epsg_3857(cudaStream_t stream, const struct dec_image *in,
                       int orig_width, int orig_height)
{
        const double src_ratio = (double) orig_width / orig_height;

        double lat_merc_top = 0;
        double lat_merc_bottom = 0;
        double lon_merc_left = 0;
        double lon_merc_right = 0;
        gcs_to_webm(in->bounds[YTOP], in->bounds[XLEFT], &lat_merc_top,
                    &lat_merc_left);
        gcs_to_webm(in->bounds[YBOTTOM], in->bounds[XRIGHT], &lat_merc_bottom,
                    &lat_merc_right);
        double dst_width = lon_merc_left - lon_merc_right;
        double dst_ratio = dst_width / dst_height;

        struct bounds src_bounds{};
        for (unsigned i = 0; i < ARR_SIZE(in->bounds); ++i) {
                src_bounds.bound[i] = (float) in->bounds[i];
        }
        struct bounds dst_bounds{};
        dst_bounds.bound[XLEFT] = (float) lon_merc_left;
        dst_bounds.bound[YTOP] = (float) lat_merc_top;
        dst_bounds.bound[XRIGHT] = (float) lon_merc_right;
        dst_bounds.bound[YBOTTOM] = (float) lat_merc_bottom;
        // struct dec_image dst_desc = *in;

        struct dec_image dst_desc = *in;
        dst_desc.width = orig_width;
        dst_desc.height = orig_height;
        if (dst_ratio >= src_ratio) {
                dst_desc.width *= dst_ratio;
        } else {
                dst_desc.height /= dst_ratio;
        }
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);

        dim3 block(16, 16);
        int width = dst_desc.width;
        int height = dst_desc.height;
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        kernel_to_web_mercator<<<grid, block, 0, stream>>>(
            in->data, ret->img.data, in->width, in->height, width, height, src_bounds, dst_bounds);

        return ret;
}

#else

template <int components, bool alpha>
static __global__ void
kernel_utm_to_web_mercator(device_projection const d_proj, const uint8_t *d_in,
                           uint8_t *d_out, uint8_t *d_out_alpha, int in_width,
                           int in_height, int out_width, int out_height,
                           struct bounds src_bounds, struct bounds dst_bounds)
{
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
                        d_out[components * (out_x + out_y * out_width) + i] = 0;
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
                d_out[components * (out_x + out_y * out_width) + i] =
                    bilinearSample(d_in + i, in_width, components, in_height,
                                   abs_pos_src_x, abs_pos_src_y);
        }
}

static bool transform_to_float(double *x, double *y,
                              OGRCoordinateTransformationH transform)
{

        constexpr double M = 20037508.342789244;
        if (OCTTransform(transform, 1, x, y, NULL)) {
                *x = (*x + M) / (2 * M);
                // why 1- ?
                *y = 1 - ((*y + M) / (2 * M));
                return true;
        }
        ERROR_MSG("Cannot tranform!\n");
        return false;
}

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
        dst_bounds->bound[XLEFT] = min(dst_bounds->bound[XLEFT], x);
        dst_bounds->bound[XRIGHT] = max(dst_bounds->bound[XRIGHT], x);
        dst_bounds->bound[YTOP] = min(dst_bounds->bound[YTOP], y);
        dst_bounds->bound[YBOTTOM] = max(dst_bounds->bound[YBOTTOM], y);
        return true;
}

static struct owned_image *utm_to_epsg_3857(struct rotate_utm_state *s,
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
        OGRSpatialReferenceH src_srs = OSRNewSpatialReference(NULL);
        OGRSpatialReferenceH dst_srs = OSRNewSpatialReference(NULL);
        OSRImportFromEPSG(src_srs, atoi(strchr(in->authority, ':') + 1));
        OSRImportFromEPSG(dst_srs, 3857);
        OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(
            src_srs, dst_srs);
        if (transform == nullptr) {
                ERROR_MSG("Cannot create transform!\n");
                return nullptr;
        }
        bool succeed = adjust_dst_bounds(XLEFT, YTOP, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XRIGHT, YTOP, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XRIGHT, YBOTTOM, &src_bounds, &dst_bounds, transform);
        succeed &= adjust_dst_bounds(XLEFT, YBOTTOM, &src_bounds, &dst_bounds, transform);
        if (!succeed) {
                return nullptr;
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

        // decrease size for GPUJPEG
        adjust_size(&dst_desc.width, &dst_desc.height, dst_desc.comp_count);

        dst_desc.alpha = output_format == OUTF_WEBP ? (unsigned char *)!NULL
                                                    : NULL;
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        snprintf(ret->img.authority, sizeof ret->img.authority, "%s", "EPSG:3857");
        for (unsigned i = 0; i < ARR_SIZE(dst_bounds.bound); ++i) {
                ret->img.bounds[i] = dst_bounds.bound[i];
        }

        dim3 block(16, 16);
        int width = dst_desc.width;
        int height = dst_desc.height;
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        decltype(kernel_utm_to_web_mercator<1, true>) *kernel = nullptr;
        if (output_format == OUTF_WEBP) {
                if (in->comp_count == 1) {
                        kernel = kernel_utm_to_web_mercator<1, true>;
                } else {
                        kernel = kernel_utm_to_web_mercator<3, true>;
                }
        } else {
                if (in->comp_count == 1) {
                        kernel = kernel_utm_to_web_mercator<1, false>;
                } else {
                        kernel = kernel_utm_to_web_mercator<3, false>;
                }
        }
        kernel<<<grid, block, 0, s->stream>>>(
            d_proj, in->data, ret->img.data, ret->img.alpha, in->width,
            in->height, width, height, src_bounds, dst_bounds);

        // Cleanup
        OCTDestroyCoordinateTransformation(transform);
        OSRDestroySpatialReference(src_srs);
        OSRDestroySpatialReference(dst_srs);

        GPU_TIMER_STOP(utm_to_epsg_3857);

        return ret;
}
#endif

struct owned_image *rotate_utm(struct rotate_utm_state *s, const struct dec_image *in)
{
#ifdef TWO_STEP_CONV
        if (in->comp_count > 1) {
                WARN_MSG("TODO: implement more than 1 channel (have %d)!\n",
                         in->comp_count);
                return nullptr;
        }

        struct owned_image *epsg4326 = to_epsg_4326(s, in);
        if (epsg4326 == nullptr) {
                return nullptr;
        }
        struct owned_image *epsg3857 = epsg_4326_to_epsg_3857(
            s->stream, &epsg4326->img, in->width, in->height);
        epsg4326->free(epsg4326);
        return epsg3857;
#else
        return utm_to_epsg_3857(s, in);
#endif
}
