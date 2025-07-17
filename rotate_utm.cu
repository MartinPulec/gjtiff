#include "rotate_utm.h"

#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#define NDEBUG
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>
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

/**
 * normalize the coordinates to 0..1 and
 * @return asoect ratio
 */
static double normalize_coords(const struct coordinate src_coords[4],
                               struct coordinate coords[4])
{
        double lat_min = 0;
        double lat_max = 0;
        double lon_min = 0;
        double lon_max = 0;

        // check if not near poles
        get_lat_lon_min_max(src_coords, &lat_min, &lat_max, &lon_min, &lon_max);
        if (lat_min < -85. || lat_max > 85.) {
                const double near_pole_pt_lat = (lat_max > 85 ? lat_max
                                                              : lat_min);
                WARN_MSG("Not normalizing areas near North/South Pole! (at "
                         "least one point at most 5° /%f°/ degrees from the "
                         "Pole)\n",
                         near_pole_pt_lat);
                return -1;
        }

        for (unsigned i = 0; i < 4; ++i) {
                double lat_rad = src_coords[i].latitude / 180. * M_PI;
                coords[i].latitude = (M_PI -
                                      log(tan((M_PI / 4.) + (lat_rad / 2.)))) /
                                     (2. * M_PI);
                double lon_rad = src_coords[i].longitude / 180. * M_PI;
                coords[i].longitude = (M_PI + lon_rad) / (2. * M_PI);
        }

        get_lat_lon_min_max(coords, &lat_min, &lat_max, &lon_min, &lon_max);

        double lat_range = lat_max - lat_min;
        double lon_range = lon_max - lon_min;

        // fprintf(stderr,
        //         "lat_min: %f lat_max: %f lon_min: %f lon_max: %f lat_range: %f "
        //         "lon_range: %f\n",
        //         lat_min, lat_max, lon_min, lon_max, lat_range, lon_range);

        for (unsigned i = 0; i < 4; ++i) {
                coords[i].latitude = (coords[i].latitude - lat_min) / lat_range;
                coords[i].longitude = (coords[i].longitude - lon_min) /
                                      lon_range;
        }

        return lon_range / lat_range;
}

/// fullfill GPUJPEG mem requirements
static void adjust_size(int *width, int *height, int comp_count) {

        enum {
                GB1 = 1LL * 1000 * 1000 * 1000,
                GJ_PER_BYTE_REQ = 20,
        };
        ssize_t threshold = mem_limit;
        if (threshold == 0) {
                threshold = MIN((ssize_t)gpu_memory / 2,
                                (ssize_t)gpu_memory - 2 * GB1);
                assert(threshold >= (ssize_t)2 * GB1);
        }
        ssize_t gj_gram_needed = (ssize_t)*width * *height * comp_count *
                                 GJ_PER_BYTE_REQ;
        if (gj_gram_needed < threshold) {
                return;
        }
        WARN_MSG(
            "[rotate] Encoding of %dx%d image would require %.2f GB GRAM (>=%g "
            "GB), downsizing ",
            *width, *height, (double)gj_gram_needed / GB1,
            (double)threshold / GB1);
        while (gj_gram_needed > threshold) {
                *width /= 2;
                *height /= 2;
                gj_gram_needed /= 4;
        }
        WARN_MSG("rotated to %.2f GB (%dx%d).\n", (double)gj_gram_needed / GB1,
                 *width, *height);
}

static void release_owned_image(struct owned_image *img) {
        CHECK_CUDA(cudaFree(img->img.data));
        free(img);
}

struct bounds {
        float bound[4];
};
using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;
static __global__ void kernel(device_projection const d_proj,device_projection const d_proj2,
                              const uint8_t *d_in, uint8_t *d_out, int in_width,
                              int in_height, int out_width, int out_height,
                              struct bounds src_bounds,
                              struct bounds dst_bounds)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (x >= out_width || y >= out_height) {
                return;
        }

        float lat_scale = dst_bounds.bound[YMAX] - dst_bounds.bound[YMIN];
        float this_lat = dst_bounds.bound[YMIN];
        this_lat += lat_scale * ((x + .5f) / out_height);

        float lon_scale = dst_bounds.bound[XMAX] - dst_bounds.bound[XMIN];
        float this_lon = dst_bounds.bound[XMIN];
        this_lon += lon_scale * ((y + .5f) / out_width);

        if (x > 0 || y > 0) return;
        cuproj::vec_2d<float> in{500000,0};
        printf("\t%f ddd  %f\n\n\n\n", this_lat, this_lon);
        printf("\t%f ddd %f\n\n\n\n", in.x, in.y);
        cuproj::vec_2d<float> out = d_proj.transform(in);
        printf("%f %f\n", out.x, out.y);
        in = d_proj2.transform(out);
        printf("%f %f\n", in.x, in.y);
        
        d_out[x + y * out_width] = d_in[x + y * in_width];
}

__global__ void example_kernel(device_projection const d_proj,
                               cuproj::vec_2d<float> const* in,
                               cuproj::vec_2d<float>* out,
                               size_t n)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = d_proj.transform(in[i]);
  }
}

void test() {
  using coordinate = cuproj::vec_2d<float>;

  // Make a projection to convert WGS84 (lat, lon) coordinates to
  // UTM zone 56S (x, y) coordinates
  auto proj = cuproj::make_projection<coordinate>("EPSG:4326", "EPSG:32756");

  // Sydney, NSW, Australia
  coordinate sydney{-33.858700, 151.214000};
  thrust::device_vector<coordinate> d_in{1, sydney};
  thrust::device_vector<coordinate> d_out(d_in.size());

  auto d_proj            = proj->get_device_projection(cuproj::direction::FORWARD);
  std::size_t block_size = 256;
  std::size_t grid_size  = (d_in.size() + block_size - 1) / block_size;
  example_kernel<<<grid_size, block_size>>>(
    d_proj, d_in.data().get(), d_out.data().get(), d_in.size());
  cudaDeviceSynchronize();
        CHECK_CUDA(cudaDeviceSynchronize());
        exit(1);
        
}

static struct owned_image *to_epsg_4326(struct rotate_utm_state *s,
                                        const struct dec_image *in)
{
        // test(); return nullptr;
        double src_ratio = (in->bounds[XMAX] - in->bounds[XMIN]) /
                           (in->bounds[YMAX] - in->bounds[YMIN]);
        double lat_top = max(in->coords[ULEFT].latitude, in->coords[URIGHT].latitude);
        double lat_bot = min(in->coords[BLEFT].latitude, in->coords[BRIGHT].latitude);
        double lon_left = min(in->coords[ULEFT].longitude, in->coords[BLEFT].longitude);
        double lon_right = max(in->coords[URIGHT].longitude, in->coords[BRIGHT].longitude);
        double dst_ratio = (lon_right - lon_left) / (lat_top - lat_bot);
        struct bounds dst_bounds;
        dst_bounds.bound[XMIN] = lon_left;
        dst_bounds.bound[YMAX] = lat_top;
        dst_bounds.bound[XMAX] = lon_right;
        dst_bounds.bound[YMIN] = lat_bot;
        struct bounds src_bounds;
        src_bounds.bound[XMIN] = in->bounds[XMIN];
        src_bounds.bound[YMAX] = in->bounds[YMAX];
        src_bounds.bound[XMAX] = in->bounds[XMAX];
        src_bounds.bound[YMIN] = in->bounds[YMIN];
        struct dec_image dst_desc = *in;
        if (dst_ratio >= src_ratio) {
                dst_desc.width = (int)(in->height * dst_ratio);
        } else {
                dst_desc.height = (int)(in->width / dst_ratio);
        }
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);

        using coordinate = cuproj::vec_2d<float>;
        auto proj = cuproj::make_projection<coordinate>(in->authority, "EPSG:4326"
                                                         );
        auto d_proj = proj->get_device_projection(cuproj::direction::FORWARD);
        auto d_proj2 = proj->get_device_projection(cuproj::direction::INVERSE);

        dim3 block(16, 16);
        int width = dst_desc.width;
        int height = dst_desc.height;
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        kernel<<<grid, block, 0, s->stream>>>(d_proj,d_proj2,
            in->data, ret->img.data, in->width, in->height, width, height, src_bounds, dst_bounds);

        return ret;
}

struct owned_image *rotate_utm(struct rotate_utm_state *s, const struct dec_image *in)
{
        struct owned_image *epsg4326 = to_epsg_4326(s, in);
        if (epsg4326 == nullptr) {
                return nullptr;
        }
        return epsg4326;

        double aSrcQuad[4][2] = {
            {0.0, 0.0},                              // Top-left
            {(double)in->width, 0},                  // Top-right
            {(double)in->width, (double)in->height}, // Bottom-right
            {0.0, (double)in->height}                // Bottom-left
        };

        struct coordinate coords[4];
        const double dst_aspect = normalize_coords(in->coords, coords);

        NppiRect oSrcROI = {0, 0, in->width, in->height};
        NppiSize oSrcSize = {in->width, in->height};

        struct owned_image *ret = (struct owned_image *) malloc(sizeof *ret);
        memcpy(&ret->img, in, sizeof *in);
        ret->free = release_owned_image;
        // keep one side as in original and upscale the other to meet dst
        // projection dimension
        const double src_aspect = (double)in->width / in->height;
        if (dst_aspect >= src_aspect) {
                ret->img.width = (int)(ret->img.height * dst_aspect);
        } else {
                ret->img.height = (int)(ret->img.width / dst_aspect);
        }
        adjust_size(&ret->img.width, &ret->img.height, in->comp_count);

        const size_t req_size = (size_t)ret->img.width * ret->img.height *
                                ret->img.comp_count;
        CHECK_CUDA(cudaMalloc((void **)&ret->img.data, req_size));

        NppiRect oDstROI = {0, 0, ret->img.width, ret->img.height};
        double aDstQuad[4][2] = {
            {coords[0].longitude * ret->img.width,
             coords[0].latitude * ret->img.height}, // Top-left
            {coords[1].longitude * ret->img.width,
             coords[1].latitude * ret->img.height}, // Top-right
            {coords[2].longitude * ret->img.width,
             coords[2].latitude * ret->img.height}, // Bottom-right
            {coords[3].longitude * ret->img.width,
             coords[3].latitude * ret->img.height}, // Bottom-left
        };

        GPU_TIMER_START(rotate, LL_DEBUG, s->stream);
        CHECK_CUDA(cudaMemsetAsync(ret->img.data, 0,
                                   (size_t)ret->img.width * ret->img.height *
                                       ret->img.comp_count,
                                   s->stream));
        const int interpolation = NPPI_INTER_LINEAR;
        if (in->comp_count == 1) {
                CHECK_NPP(nppiWarpPerspectiveQuad_8u_C1R_Ctx(
                    in->data, oSrcSize, in->width, oSrcROI, aSrcQuad,
                    ret->img.data, ret->img.width, oDstROI, aDstQuad,
                    interpolation, s->nppStreamCtx));
        } else {
                assert(in->comp_count == 3);
                CHECK_NPP(nppiWarpPerspectiveQuad_8u_C3R_Ctx(
                    in->data, oSrcSize, 3 * in->width, oSrcROI, aSrcQuad,
                    ret->img.data, 3 * ret->img.width, oDstROI, aDstQuad,
                    interpolation, s->nppStreamCtx));
        }
        GPU_TIMER_STOP(rotate);

        return ret;
}
