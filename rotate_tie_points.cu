#include "rotate_tie_points.h"

#include <cassert>              // for assert
#include <cstdlib>              // for abort, calloc, free
#include <ogr_srs_api.h>

// #include "cuda_common.cuh"
#include "defs.h"
#include "nppdefs.h"
#include "rotate_utm.h" // for transform_to_float
#include "utils.h"              // for ERROR_MSG

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

template <int components, bool alpha>
static __global__ void
kernel_tie_points(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_out_alpha,
                  int in_width, int in_height, int out_width, int out_height,
                  struct bounds dst_bounds, struct tie_points tie_points)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out_width || out_y >= out_height) {
                return;
        }

        float y_scale = dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP];
        float this_y = dst_bounds.bound[YTOP];
        this_y += y_scale * ((out_y + .5f) / out_height) ;

        float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
        float this_x = dst_bounds.bound[XLEFT];
        this_x += x_scale * ((out_x + .5f) / out_width);

        int val = 0;

        unsigned grid_width = tie_points.grid_width;
        unsigned grid_height = tie_points.count / tie_points.grid_width;
        for (unsigned grid_x = 0; grid_x < grid_width - 1; ++grid_x) {
                for (unsigned grid_y = 0; grid_y < grid_height - 1; ++grid_y) {
                        const struct tie_point *a = &tie_points.points[grid_x + (grid_y * grid_width)];
                        const struct tie_point *b = &tie_points.points[(grid_x + 1) + (grid_y * grid_width)];
                        const struct tie_point *c = &tie_points.points[(grid_x + 1)+ ((grid_y + 1) * grid_width)];
                        const struct tie_point *d = &tie_points.points[grid_x + ((grid_y + 1) * grid_width)];
                        // struct tie_point aa= {.webx =0.537795, .weby = 0.34 };
                        // struct tie_point bb= {.webx =0.549097, .weby = 0.334925 };
                        // struct tie_point cc= {.webx =0.549097, .weby = 0.343166};
                        // struct tie_point dd= {.webx =0.537795, .weby = 0.343166};
                        // a = &aa;
                        // b = &bb;
                        // c = &cc;
                        // d = &dd;

                        double cross1 = (b->webx - a->webx) * (this_y - a->weby) - (b->weby - a->weby) * (this_x - a->webx);
                        double cross2 = (c->webx - b->webx) * (this_y - b->weby) - (c->weby - b->weby) * (this_x - b->webx);
                        double cross3 = (d->webx - c->webx) * (this_y - c->weby) - (d->weby - c->weby) * (this_x - c->webx);
                        double cross4 = (a->webx - d->webx) * (this_y - d->weby) - (a->weby - d->weby) * (this_x - d->webx);

                        bool all_positive = (cross1 >= 0 and cross2 >= 0 and cross3 >= 0 and cross4 >= 0);
                        bool all_negative = (cross1 <= 0 and cross2 <= 0 and cross3 <= 0 and cross4 <= 0);
                        if (all_positive || all_negative ) {
                                val = 255;
                                
                        }
                }
        }


        // dummy white
        for (int i = 0; i < components; ++i) {
                d_out[(components * (out_x + out_y * out_width)) + i] = val;
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
        CHECK_CUDA(cudaStreamSynchronize(s->stream));
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
        dim3 block(16, 16);
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
