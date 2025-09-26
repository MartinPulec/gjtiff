#include "rotate_tie_points.h"

#include <cassert>              // for assert
#include <cstdlib>              // for abort, calloc, free
#include <ogr_srs_api.h>

// #include "cuda_common.cuh"
#include "cuda_common.cuh"  // for bilinearSample
#include "defs.h"
#include "nppdefs.h"
#include "rotate_utm.h" // for transform_to_float
#include "utils.h"              // for ERROR_MSG

enum {
        THREAD_X_COUNT = 16,
        THREAD_Y_COUNT = 16,
        THREAD_COUNT = THREAD_X_COUNT * THREAD_Y_COUNT,
        WARP_SZ = 32,
};

struct rotate_tie_points_state {
        cudaStream_t stream;
        OGRCoordinateTransformationH transform;

        struct tie_point *d_tie_points;
        unsigned tie_point_allocated_count;
};

struct rotate_tie_points_state *rotate_tie_points_init(cudaStream_t stream)
{
        auto *s = new rotate_tie_points_state{};
        assert(s != nullptr);
        s->stream = stream;

        OGRSpatialReferenceH src_srs = OSRNewSpatialReference(nullptr);
        OGRSpatialReferenceH dst_srs = OSRNewSpatialReference(nullptr);
        OSRImportFromEPSG(src_srs, EPSG_WGS_84);
        OSRImportFromEPSG(dst_srs, EPSG_WEB_MERCATOR);
        s->transform = OCTNewCoordinateTransformation(src_srs, dst_srs);
        OSRDestroySpatialReference(src_srs);
        OSRDestroySpatialReference(dst_srs);
        if (s->transform == nullptr) {
                ERROR_MSG("Cannot create transform!\n");
                return nullptr;
        }

        return s;
}

void rotate_tie_points_destroy(struct rotate_tie_points_state *s)
{
        if (s == nullptr) {
                return;
        }
        OCTDestroyCoordinateTransformation(s->transform);
        CHECK_CUDA(cudaFree(s->d_tie_points));
        delete s;
}

struct bounds {
        float bound[4];
};

static __global__ void kernel_tie_points(unsigned tie_point_count,
                                         struct tie_point *tie_points)
{
        unsigned pos = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (pos >= tie_point_count) {
                return;
        }
        double lat_rad = tie_points[pos].lat / 180. * M_PI;
        double lon_rad = tie_points[pos].lon / 180. * M_PI;
        tie_points[pos].weby = (M_PI - log(tan((M_PI / 4.) + (lat_rad / 2.)))) / (2. * M_PI);
        tie_points[pos].webx = (M_PI + lon_rad) / (2. * M_PI);
        // printf("%f %f\n", tie_points[pos].webx , tie_points[pos].weby);
}

enum { GRID_B_X_MIN, GRID_B_X_MAX, GRID_B_Y_MIN, GRID_B_Y_MAX, GRID_B_COUNT };

static __device__ bool
get_bounding_tie_points(const struct tie_point *tie_bounds[4], float this_x,
                        float this_y, const struct tie_points &tie_points,
                        const int grid_bounds[GRID_B_COUNT])
{
        int grid_width = tie_points.grid_width;
        for (int grid_y = grid_bounds[GRID_B_Y_MIN]; grid_y <= grid_bounds[GRID_B_Y_MAX]; ++grid_y) {
                for (int grid_x = grid_bounds[GRID_B_X_MIN]; grid_x <= grid_bounds[GRID_B_X_MAX]; ++grid_x) {
                        // clang-format off
                        const struct tie_point *a = &tie_points.points[grid_x + (grid_y * grid_width)];
                        const struct tie_point *b = &tie_points.points[(grid_x + 1) + (grid_y * grid_width)];
                        const struct tie_point *c = &tie_points.points[(grid_x + 1)+ ((grid_y + 1) * grid_width)];
                        const struct tie_point *d = &tie_points.points[grid_x + ((grid_y + 1) * grid_width)];

                        float cross1 = (b->webx - a->webx) * (this_y - a->weby) - (b->weby - a->weby) * (this_x - a->webx);
                        float cross2 = (c->webx - b->webx) * (this_y - b->weby) - (c->weby - b->weby) * (this_x - b->webx);
                        float cross3 = (d->webx - c->webx) * (this_y - c->weby) - (d->weby - c->weby) * (this_x - c->webx);
                        float cross4 = (a->webx - d->webx) * (this_y - d->weby) - (a->weby - d->weby) * (this_x - d->webx);
                        // clang-format on

                        bool all_positive = (cross1 >= 0 and cross2 >= 0 and
                                             cross3 >= 0 and cross4 >= 0);
                        bool all_negative = (cross1 <= 0 and cross2 <= 0 and
                                             cross3 <= 0 and cross4 <= 0);
                        if (all_positive || all_negative) {
                                tie_bounds[0] = a;
                                tie_bounds[1] = b;
                                tie_bounds[2] = c;
                                tie_bounds[3] = d;
                                return true;
                        }
                }
        }
        return false;
}

/**
 * compute region of interest of tie-points for current thread block in parallel
 *
 * In every thread, take 4 bounding points of current thread block and try to find
 * used tie point regions.
 *
 * @returns false if the whole thread block lies out-of-bounds of the source img
 */
static __device__ bool set_grid_bounds(const struct tie_points &tie_points,
                                       struct bounds &dst_bounds,
                                       float out_width, float out_height,
                                       int grid_bounds[4])
{
        int thread_id = (blockDim.x * threadIdx.y) + threadIdx.x;
        int thread_count = blockDim.x * blockDim.y;

        int base_x = blockIdx.x * blockDim.x;
        int base_y = blockIdx.y * blockDim.y;

        // block bounding pixels
        int x[4];
        int y[4];
        x[0] = base_x;
        y[0] = base_y;
        x[1] = base_x + blockDim.x - 1;
        y[1] = base_y;
        x[2] = base_x + blockDim.x - 1;
        y[2] = base_y + blockDim.y - 1;
        x[3] = base_x;
        y[3] = base_y + blockDim.y - 1;

        float merc_x[4];
        float merc_y[4];

        const float y_scale = dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP];
        const float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];

        for (int i = 0; i < 4; ++i) {
                merc_x[i] = dst_bounds.bound[XLEFT] +
                            x_scale * (((float)x[i] + .5f) / out_width);
                merc_y[i] = dst_bounds.bound[YTOP] +
                            y_scale * (((float)y[i] + .5f) / out_height);
        }

        int grid_x_min = INT_MAX;
        int grid_x_max = -1;
        int grid_y_min = INT_MAX;
        int grid_y_max = -1;

        int grid_width = tie_points.grid_width;
        int grid_height = tie_points.count / tie_points.grid_width;
        for (unsigned i = thread_id; i < (grid_width - 1) * (grid_height - 1);
             i += thread_count) {
                int grid_x = i % (tie_points.grid_width - 1);
                int grid_y = i / (tie_points.grid_width - 1);

                // clang-format off
                const struct tie_point *a = &tie_points.points[grid_x + (grid_y * grid_width)];
                const struct tie_point *b = &tie_points.points[(grid_x + 1) + (grid_y * grid_width)];
                const struct tie_point *c = &tie_points.points[(grid_x + 1)+ ((grid_y + 1) * grid_width)];
                const struct tie_point *d = &tie_points.points[grid_x + ((grid_y + 1) * grid_width)];

                for (int i = 0; i < 4; ++i) {
                        float this_x = merc_x[i];
                        float this_y = merc_y[i];

                        float cross1 = (b->webx - a->webx) * (this_y - a->weby) - (b->weby - a->weby) * (this_x - a->webx);
                        float cross2 = (c->webx - b->webx) * (this_y - b->weby) - (c->weby - b->weby) * (this_x - b->webx);
                        float cross3 = (d->webx - c->webx) * (this_y - c->weby) - (d->weby - c->weby) * (this_x - c->webx);
                        float cross4 = (a->webx - d->webx) * (this_y - d->weby) - (a->weby - d->weby) * (this_x - d->webx);
                        // clang-format on

                        bool all_positive = (cross1 >= 0 and cross2 >= 0 and
                                             cross3 >= 0 and cross4 >= 0);
                        bool all_negative = (cross1 <= 0 and cross2 <= 0 and
                                             cross3 <= 0 and cross4 <= 0);
                        if (all_positive || all_negative) {
                                grid_x_min = min(grid_x_min, grid_x);
                                grid_x_max = max(grid_x_max, grid_x);
                                grid_y_min = min(grid_y_min, grid_y);
                                grid_y_max = max(grid_y_max, grid_y);
                        }
                }
        }

        // finally do the tie point ROI reduction
#if __CUDA_ARCH__ >= 800
        __shared__ int gb[THREAD_COUNT / WARP_SZ][GRID_B_COUNT];
        const auto all_x_min = __reduce_min_sync(-1U, grid_x_min);
        const auto all_x_max = __reduce_max_sync(-1U, grid_x_max);
        const auto all_y_min = __reduce_min_sync(-1U, grid_y_min);
        const auto all_y_max = __reduce_max_sync(-1U, grid_y_max);
        __syncthreads();

        if ((thread_id % WARP_SZ) == 0) {
                for (int i = 0; i < 4; ++i) {
                        gb[thread_id / WARP_SZ][GRID_B_X_MIN] = all_x_min;
                        gb[thread_id / WARP_SZ][GRID_B_X_MAX] = all_x_max;
                        gb[thread_id / WARP_SZ][GRID_B_Y_MIN] = all_y_min;
                        gb[thread_id / WARP_SZ][GRID_B_Y_MAX] = all_y_max;
                }
        }
        __syncthreads();

        if (thread_id == 0) {
                grid_bounds[GRID_B_X_MIN] = INT_MAX;
                grid_bounds[GRID_B_X_MAX] = -1;
                grid_bounds[GRID_B_Y_MIN] = INT_MAX;
                grid_bounds[GRID_B_Y_MAX] = -1;
                for (unsigned i = 0; i < sizeof gb / sizeof gb[0]; ++i) {
                        grid_bounds[GRID_B_X_MIN] = min(gb[i][GRID_B_X_MIN], grid_bounds[GRID_B_X_MIN]);
                        grid_bounds[GRID_B_X_MAX] = max(gb[i][GRID_B_X_MAX], grid_bounds[GRID_B_X_MAX]);
                        grid_bounds[GRID_B_Y_MIN] = min(gb[i][GRID_B_Y_MIN], grid_bounds[GRID_B_Y_MIN]);
                        grid_bounds[GRID_B_Y_MAX] = max(gb[i][GRID_B_Y_MAX], grid_bounds[GRID_B_Y_MAX]);
                }
        }
        __syncthreads();

#else // __CUDA_ARCH__ <= 800
        __shared__ int gb[THREAD_X_COUNT * THREAD_Y_COUNT][GRID_B_COUNT];
        gb[thread_id][GRID_B_X_MIN] = grid_x_min;
        gb[thread_id][GRID_B_X_MAX] = grid_x_max;
        gb[thread_id][GRID_B_Y_MIN] = grid_y_min;
        gb[thread_id][GRID_B_Y_MAX] = grid_y_max;
        __syncthreads();

        if (thread_id == 0) {
                grid_bounds[GRID_B_X_MIN] = INT_MAX;
                grid_bounds[GRID_B_X_MAX] = -1;
                grid_bounds[GRID_B_Y_MIN] = INT_MAX;
                grid_bounds[GRID_B_Y_MAX] = -1;

                for (int i = 0; i < THREAD_COUNT; ++i) {
                        grid_bounds[GRID_B_X_MIN] = min(gb[i][GRID_B_X_MIN], grid_bounds[GRID_B_X_MIN]);
                        grid_bounds[GRID_B_X_MAX] = max(gb[i][GRID_B_X_MAX], grid_bounds[GRID_B_X_MAX]);
                        grid_bounds[GRID_B_Y_MIN] = min(gb[i][GRID_B_Y_MIN], grid_bounds[GRID_B_Y_MIN]);
                        grid_bounds[GRID_B_Y_MAX] = max(gb[i][GRID_B_Y_MAX], grid_bounds[GRID_B_Y_MAX]);
                }
        }
#endif
        __syncthreads();
        return grid_bounds[GRID_B_X_MIN] != INT_MAX;
}

template <int components, bool alpha>
static __global__ void
kernel_tie_points(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_out_alpha,
                  int in_width, int in_height, unsigned out_width, unsigned out_height,
                  struct bounds dst_bounds, struct tie_points tie_points)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        __shared__ int grid_bounds[GRID_B_COUNT];
        if (!set_grid_bounds(tie_points, dst_bounds, (float)out_width,
                             (float)out_height, grid_bounds)) {
                return;
        }

        // do the return after set_grid_bounds -> we need all threads for the fn
        if (out_x >= out_width || out_y >= out_height) {
                return;
        }

        float y_scale = dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP];
        float this_y = dst_bounds.bound[YTOP] +
                       (y_scale * (((float)out_y + .5F) / (float)out_height));

        float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
        float this_x = dst_bounds.bound[XLEFT] +
                       (x_scale * (((float)out_x + .5F) / (float)out_width));

        const struct tie_point *tie_bounds[4];
        if (!get_bounding_tie_points(tie_bounds, this_x, this_y, tie_points,
                                     grid_bounds)) {
                for (int i = 0; i < components; ++i) {
                        d_out[(components * (out_x + out_y * out_width)) + i] =
                            0;
                }
                if (alpha) {
                        d_out_alpha[out_x + (out_y * out_width)] = 0;
                }
                return;
        }

        if (alpha) {
                d_out_alpha[out_x + (out_y * out_width)] = 255;
        }

        // clang-format off
        float dists[4]; // from lines AB,BC,CD,DA (letters are tie points)
        for (int i = 0; i < 4; ++i) {
                int m = i;
                int n = (i + 1) % 4;
                dists[i] = fabsf(((tie_bounds[n]->webx - tie_bounds[m]->webx) * (this_y - tie_bounds[m]->weby)) -
                                 ((tie_bounds[n]->weby - tie_bounds[m]->weby) * (this_x - tie_bounds[m]->webx))) /
                           sqrtf(powf(tie_bounds[n]->webx - tie_bounds[m]->webx, 2) +
                                 powf(tie_bounds[n]->weby - tie_bounds[m]->weby, 2));
        }
        float dist_y = dists[0] + dists[2]; // distance of current point from line AB + CD
        float src_y = (((float) tie_bounds[0]->y * (dist_y - dists[0])) +
                       ((float) tie_bounds[2]->y * (dist_y - dists[2]))) / dist_y;
        float dist_x = dists[1] + dists[3];
        float src_x = (((float) tie_bounds[1]->x * (dist_x - dists[1])) +
                       ((float) tie_bounds[3]->x * (dist_x - dists[3]))) / dist_x;
        // clang-format on

        // if (out_x == out_width/2 && out_y == out_height/2) {
        //         printf("%f %f\n", this_x, this_y);
        //         for (int i = 0; i< 4;++i) {
        //                 printf("%hu %hu %f %f\n", tie_bounds[i]->x,tie_bounds[i]->y, tie_bounds[i]->webx,tie_bounds[i]->weby );
        //         }
        //         printf("%f %f\n", dists[0], dists[2]);
        //         printf("%f %f\n", src_x, src_y);
        // }

        for (int i = 0; i < components; ++i) {
                d_out[(components * (out_x + out_y * out_width)) + i] =
                    bilinearSample(d_in + i, in_width, components, in_height,
                                   src_x, src_y);
        }
}

struct owned_image *rotate_tie_points(struct rotate_tie_points_state *s, const struct dec_image *in)
{
        if (in->e3857_sug_w == 0 || in->e3857_sug_h == 0) {
                WARN_MSG("Suggested size set to 0, skipping rotate_tie_points...\n");
                return nullptr;
        }
        GPU_TIMER_START(rotate_geotff, LL_DEBUG, s->stream);
        struct dec_image dst_desc = *in;
        dst_desc.width = in->e3857_sug_w;
        dst_desc.height = in->e3857_sug_h;

        dst_desc.alpha = output_format == OUTF_WEBP ? (unsigned char *)1
                                                    : nullptr;
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        snprintf(ret->img.authority, sizeof ret->img.authority, "EPSG:%d",
                 EPSG_WEB_MERCATOR);

        double x = ret->img.bounds[YTOP];
        double y = ret->img.bounds[XLEFT];
        transform_to_float(&x, &y, s->transform);
        ret->img.bounds[XLEFT] = x;
        ret->img.bounds[YTOP] = y;
        x = ret->img.bounds[YBOTTOM];
        y = ret->img.bounds[XRIGHT];
        transform_to_float(&x, &y, s->transform);
        ret->img.bounds[XRIGHT] = x;
        ret->img.bounds[YBOTTOM] = y;

        assert(in->tie_points.points != nullptr);

        if (in->tie_points.count > s->tie_point_allocated_count) {
                CHECK_CUDA(cudaFreeAsync(s->d_tie_points, s->stream));
                CHECK_CUDA(cudaMallocAsync((void **)&s->d_tie_points,
                                           in->tie_points.count *
                                               sizeof *s->d_tie_points,
                                           s->stream));
                s->tie_point_allocated_count = in->tie_points.count;
        }
        CHECK_CUDA(cudaMemcpyAsync(s->d_tie_points, in->tie_points.points,
                                   in->tie_points.count *
                                       sizeof *s->d_tie_points,
                                   cudaMemcpyDefault, s->stream));

        kernel_tie_points<<<
            dim3(256), dim3((in->tie_points.count + 255) / 256), 0, s->stream>>>(
            in->tie_points.count, s->d_tie_points);
        CHECK_CUDA(cudaGetLastError());

        struct bounds dst_bounds{};
        for (unsigned i = 0; i < ARR_SIZE(dst_bounds.bound); ++i) {
                dst_bounds.bound[i] = (float) ret->img.bounds[i];
        }
        dim3 block(THREAD_X_COUNT, THREAD_Y_COUNT);
        int dst_width = dst_desc.width;
        int dst_height = dst_desc.height;
        dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

        auto *kernel_alpha = in->comp_count == 1 ? kernel_tie_points<1, true>
                                                 : kernel_tie_points<3, true>;
        auto *kernel_wo_alpha = in->comp_count == 1
                                    ? kernel_tie_points<1, false>
                                    : kernel_tie_points<3, false>;
        auto *kernel = output_format == OUTF_WEBP ? kernel_alpha
                                                  : kernel_wo_alpha;

        struct tie_points d_tie_points = in->tie_points;
        d_tie_points.points = s->d_tie_points;

        kernel<<<grid, block, 0, s->stream>>>(
            in->data, ret->img.data, ret->img.alpha, in->width, in->height,
            dst_width, dst_height, dst_bounds, d_tie_points);
        CHECK_CUDA(cudaGetLastError());

        GPU_TIMER_STOP(rotate_geotff);

        return ret;
}
