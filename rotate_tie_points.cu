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
};

struct rotate_tie_points_state *rotate_tie_points_init(cudaStream_t stream)
{
        struct rotate_tie_points_state *s = (struct rotate_tie_points_state *)calloc(
            1, sizeof *s);
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
        free(s);
}
// template <int components, bool alpha>
// static __global__ void
// kernel_utm_to_web_mercator(device_projection const d_proj, const uint8_t *d_in,
//                            uint8_t *d_out, uint8_t *d_out_alpha, int in_width,
//                            int in_height, int out_width, int out_height,
//                            struct bounds src_bounds, struct bounds dst_bounds)
// {
//         int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
//         int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

//         if (out_x >= out_width || out_y >= out_height) {
//                 return;
//         }

//         float y_scale = dst_bounds.bound[YBOTTOM] - dst_bounds.bound[YTOP];
//         float this_y = dst_bounds.bound[YTOP];
//         this_y += y_scale * ((out_y + .5f) / out_height);

//         float x_scale = dst_bounds.bound[XRIGHT] - dst_bounds.bound[XLEFT];
//         float this_x = dst_bounds.bound[XLEFT];
//         this_x += x_scale * ((out_x + .5f) / out_width);

//         // transformace
//         float pos_wgs84_lon = 360. * this_x - 180.; // lambda
//         float t = (float) M_PI * (1. - 2. *  this_y);
//         float fi_rad = 2 * atanf(powf(M_E, t)) - (M_PI / 2);
//         float pos_wgs84_lat = fi_rad * 180. / M_PI;

//         cuproj::vec_2d<float> pos_wgs84{pos_wgs84_lat, pos_wgs84_lon};
//         cuproj::vec_2d<float> pos_utm = d_proj.transform(pos_wgs84);
//         pos_utm = d_proj.transform(pos_wgs84);

//         float rel_pos_src_x = (pos_utm.x - src_bounds.bound[XLEFT]) /
//                               (src_bounds.bound[XRIGHT] - src_bounds.bound[XLEFT]);
//         float rel_pos_src_y = (pos_utm.y - src_bounds.bound[YTOP]) /
//                               (src_bounds.bound[YBOTTOM] - src_bounds.bound[YTOP]);

//         if (rel_pos_src_x < 0 || rel_pos_src_x > 1 ||
//             rel_pos_src_y < 0 || rel_pos_src_y > 1) {
//                 for (int i = 0; i < components; ++i) {
//                         d_out[components * (out_x + out_y * out_width) + i] = 0;
//                 }
//                 if (alpha) {
//                         d_out_alpha[out_x + (out_y * out_width)] = 0;
//                 }
//                 return;
//         }
//         if (alpha) {
//                 d_out_alpha[out_x + (out_y * out_width)] = 255;
//         }
//         // if (out_y == 0) {
//         //         printf("%f %f\n" , rel_pos_src_x, rel_pos_src_y);
//         // }

//         float abs_pos_src_x = rel_pos_src_x * in_width;
//         float abs_pos_src_y = rel_pos_src_y * in_height;

//         for (int i = 0; i < components; ++i) {
//                 d_out[components * (out_x + out_y * out_width) + i] =
//                     bilinearSample(d_in + i, in_width, components, in_height,
//                                    abs_pos_src_x, abs_pos_src_y);
//         }
// }

struct owned_image *rotate_tie_points(struct rotate_tie_points_state *s, const struct dec_image *in)
{
        if (in->e3857_sug_w == 0 || in->e3857_sug_h == 0) {
                WARN_MSG("Suggested size set to 0, skipping rotate_tie_points...\n");
                return nullptr;
        }
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

        abort();

        // dim3 block(16, 16);
        // int width = dst_desc.width;
        // int height = dst_desc.height;
        // dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        // decltype(kernel_utm_to_web_mercator<1, true>) *kernel = nullptr;
        // if (output_format == OUTF_WEBP) {
        //         if (in->comp_count == 1) {
        //                 kernel = kernel_utm_to_web_mercator<1, true>;
        //         } else {
        //                 kernel = kernel_utm_to_web_mercator<3, true>;
        //         }
        // } else {
        //         if (in->comp_count == 1) {
        //                 kernel = kernel_utm_to_web_mercator<1, false>;
        //         } else {
        //                 kernel = kernel_utm_to_web_mercator<3, false>;
        //         }
        // }
        // kernel<<<grid, block, 0, s->stream>>>(
        //     d_proj, in->data, ret->img.data, ret->img.alpha, in->width,
        //     in->height, width, height, src_bounds, dst_bounds);


        return ret;
}
