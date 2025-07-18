#include "rotate_utm.h"

#include <algorithm>
#include <assert.h>
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
#include <math.h>
#include <npp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

// Device function: bilinear sample at (x, y) in [0..W) × [0..H)
__device__ uint8_t bilinearSample(
    const uint8_t* src,
    int W, int H,
    float x, float y)
{
    // Compute integer bounds
    int x0 = int(floorf(x));
    int y0 = int(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // // Clamp to image edges - doesn't happen
    // x0 = max(0, min(x0, W - 1));
    // y0 = max(0, min(y0, H - 1));
    // x1 = max(0, min(x1, W - 1));
    // y1 = max(0, min(y1, H - 1));

    // Fetch four neighbors
    float I00 = src[y0 * W + x0];
    float I10 = src[y0 * W + x1];
    float I01 = src[y1 * W + x0];
    float I11 = src[y1 * W + x1];

    // fractional part
    float dx = x - float(x0);
    float dy = y - float(y0);

    // interpolate in x direction
    float a = I00 + dx * (I10 - I00);
    float b = I01 + dx * (I11 - I01);

    // interpolate in y direction
    return a + dy * (b - a);
}


struct bounds {
        float bound[4];
};
using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;
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
        if (out_y == 0) {
                printf("%f\n", dst_bounds.bound[YBOTTOM]);
                printf("HERE! %d %f %f %f %f\n" , out_x, this_lat, this_lon, pos_utm.x, pos_utm.y);
                printf("%f %f\n" , rel_pos_src_x, rel_pos_src_y);
        }

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

        float y_scale = dst_bounds.bound[YTOP] - dst_bounds.bound[YBOTTOM];
        float this_y = dst_bounds.bound[YBOTTOM];
        this_y += y_scale * ((out_y + .5f) / out_height);

        float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
        float this_x = dst_bounds.bound[XLEFT];
        this_x += x_scale * ((out_x + .5f) / out_width);

        // transformace
        float pos_wgs84_lon = 360. * this_x - 180.; // lambda
        float t = (float) M_PI * (1. - 2. *  this_y);
        float fi_rad = 2 * atanf(powf(M_E, t)) - (M_PI / 2);
        float pos_wgs84_lat = fi_rad * 180. / M_PI;

        float rel_pos_src_x = (pos_wgs84_lon - src_bounds.bound[XLEFT]) / (src_bounds.bound[XRIGHT] - src_bounds.bound[XLEFT]);
        float rel_pos_src_y = (pos_wgs84_lat - src_bounds.bound[YBOTTOM]) / (src_bounds.bound[YTOP] - src_bounds.bound[YBOTTOM]);

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

        double lat_rad_top = in->bounds[YTOP] / 180. * M_PI;
        double lat_rad_bottom = in->bounds[YBOTTOM] / 180. * M_PI;
        double lat_merc_top = (M_PI -
                               log(tan((M_PI / 4.) + (lat_rad_top / 2.)))) /
                              (2. * M_PI);
        double lat_merc_bottom = (M_PI - log(tan((M_PI / 4.) +
                                                 (lat_rad_bottom / 2.)))) /
                                 (2. * M_PI);
        double dst_height = lat_merc_top - lat_merc_bottom;
        double lon_rad_left = in->bounds[XLEFT] / 180. * M_PI;
        double lon_rad_right = in->bounds[XRIGHT] / 180. * M_PI;
        double lon_merc_left  = (M_PI + lon_rad_left) / (2. * M_PI);
        double lon_merc_right  = (M_PI + lon_rad_right) / (2. * M_PI);
        double dst_width = lon_merc_left - lon_merc_right;
        double dst_ratio = dst_width / dst_height;

        struct bounds src_bounds{};
        for (unsigned i = 0; i < ARR_SIZE(in->bounds); ++i) {
                src_bounds.bound[i] = (float) in->bounds[i];
        }
        struct bounds dst_bounds{};
        dst_bounds.bound[XLEFT] = (float) lon_merc_left;
        dst_bounds.bound[YTOP] = (float) lat_merc_bottom;
        dst_bounds.bound[XRIGHT] = (float) lon_merc_right;
        dst_bounds.bound[YBOTTOM] = (float) lat_merc_top;
        // struct dec_image dst_desc = *in;

        struct dec_image dst_desc = *in;
        if (dst_ratio >= src_ratio) {
                dst_desc.width = (int)(in->height * dst_ratio);
        } else {
                dst_desc.height = (int)(in->width / dst_ratio);
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

struct owned_image *rotate_utm(struct rotate_utm_state *s, const struct dec_image *in)
{
        struct owned_image *epsg4326 = to_epsg_4326(s, in);
        if (epsg4326 == nullptr) {
                return nullptr;
        }
        struct owned_image *epsg3857 = epsg_4326_to_epsg_3857(
            s->stream, &epsg4326->img, in->width, in->height);
        epsg4326->free(epsg4326);
        return epsg3857;
}
