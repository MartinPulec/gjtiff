#include "rotate.h"

#include <assert.h>
#include <cuda_runtime.h>
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
#include "utils.h"

#ifdef NPP_NEW_API
#define CONTEXT , s->nppStreamCtx
#else
#define CONTEXT
#endif

extern long long mem_limit; // defined in main.c

struct rotate_state {
        cudaStream_t stream;
#ifdef NPP_NEW_API
        NppStreamContext nppStreamCtx;
#endif
};

struct rotate_state *rotate_init(cudaStream_t stream)
{
        struct rotate_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

#ifdef NPP_NEW_API
        init_npp_context(&s->nppStreamCtx, stream);
#endif

        return s;
}

void rotate_destroy(struct rotate_state *s)
{
        if (s == NULL) {
                return;
        }
        free(s);
}

void get_lat_lon_min_max(struct coordinate coords[4], double *lat_min,
                         double *lat_max, double *lon_min, double *lon_max)
{
        *lat_min = 1e6;
        *lat_max = -1e6;
        *lon_min = 1e6;
        *lon_max = -1e6;

        for (unsigned i = 0; i < 4; ++i) {
                if (coords[i].latitude < *lat_min) {
                        *lat_min = coords[i].latitude;
                }
                if (coords[i].latitude > *lat_max) {
                        *lat_max = coords[i].latitude;
                }
                if (coords[i].longitude < *lon_min) {
                        *lon_min = coords[i].longitude;
                }
                if (coords[i].longitude > *lon_max) {
                        *lon_max = coords[i].longitude;
                }
        }
}

/**
 * normalize the coordinates to 0..1 and
 * @return asoect ratio
 */
static double normalize_coords_intrn(const struct coordinate src_coords[4],
                                     struct coordinate coords[4],
                                     bool second_run)
{
        double lat_min = 0;
        double lat_max = 0;
        double lon_min = 0;
        double lon_max = 0;

        enum {
                LAT_OFF = 360, // val that added is inert to cos()
                LON_OFF = 180,
        };

        if (!second_run) {
                memcpy(coords, src_coords, 4 * sizeof(struct coordinate));
                // make the coordinates positive:
                // - lat - 270-450
                // - lon - 0-360
                for (unsigned i = 0; i < 4; ++i) {
                        coords[i].latitude += LAT_OFF;
                        coords[i].longitude += LON_OFF;
                }
        }
        get_lat_lon_min_max(coords, &lat_min, &lat_max, &lon_min, &lon_max);

        // handle longitude 179->-179 transition (eastern to western
        // hemishpere); zero is now shifted to 180
        if (lon_min < LON_OFF && lon_max >= LON_OFF) {
                for (unsigned i = 0; i < 4; ++i) {
                        if (coords[i].longitude < LON_OFF) {
                                coords[i].longitude += 360.0;
                        }
                }
                assert(!second_run);
                return normalize_coords_intrn(src_coords, coords, true);
        }
        if (lat_min < LAT_OFF - 85 || lat_max > LAT_OFF + 85) {
                const double near_pole_pt_lat =
                    (lat_max > LAT_OFF + 85 ? lat_max : lat_min) - LAT_OFF;
                WARN_MSG("Not normalizing areas near North/South Pole! (at "
                         "least one point at most 5° /%f°/ degrees from the "
                         "Pole)\n", near_pole_pt_lat);
                return -1;
        }

        double lat_range = lat_max - lat_min;
        double lon_range = lon_max - lon_min;

        for (unsigned i = 0; i < 4; ++i) {
                coords[i].latitude = 1.0 -
                                     (coords[i].latitude - lat_min) / lat_range;
                coords[i].longitude = (coords[i].longitude - lon_min) /
                                      lon_range;
        }

        double lat_mean = lat_min + ((lat_max - lat_min) / 2);
        static_assert(LAT_OFF % 360 == 0);
        double lon_lat_ratio = cos(M_PI * lat_mean / 180.0);

        return (lon_range / lat_range) * lon_lat_ratio;
}

static double normalize_coords(const struct coordinate src_coords[4], struct coordinate dst_coords[4]) {
        return normalize_coords_intrn(src_coords, dst_coords, false);
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

static struct owned_image *take_ownership(const struct dec_image *in)
{
        struct owned_image *ret = malloc(sizeof *ret);
        memcpy(&ret->img, in, sizeof *in);
        const size_t size = (size_t) in->width * in->height * in->comp_count;
        CHECK_CUDA(cudaMalloc((void **)&ret->img.data, size));
        CHECK_CUDA(
            cudaMemcpy(ret->img.data, in->data, size, cudaMemcpyDefault));
        ret->free = release_owned_image;
        return ret;
}

struct owned_image *rotate(struct rotate_state *s, const struct dec_image *in)
{
        if (s == NULL) {
                return take_ownership(in);
        }
        if (!in->coords_set) {
                WARN_MSG("Coordinates not set, not normalizing image...\n");
                return take_ownership(in);
        }

#ifndef NPP_NEW_API
        if (nppGetStream() != s->stream) {
                nppSetStream(s->stream);
        }
#endif

        double aSrcQuad[4][2] = {
            {0.0, 0.0},              // Top-left
            {in->width, 0},          // Top-right
            {in->width, in->height}, // Bottom-right
            {0.0, in->height}        // Bottom-left
        };

        struct coordinate coords[4];
        const double dst_aspect = normalize_coords(in->coords, coords);
        if (dst_aspect == -1) {
                DEBUG_MSG("Near North/South pole - not rotating\n");
                return take_ownership(in);
        }
        assert(dst_aspect > 0);

        NppiRect oSrcROI = {0, 0, in->width, in->height};
        NppiSize oSrcSize = {in->width, in->height};

        struct owned_image *ret = malloc(sizeof *ret);
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
                CHECK_NPP(NPP_CONTEXTIZE(nppiWarpPerspectiveQuad_8u_C1R)(
                    in->data, oSrcSize, in->width, oSrcROI, aSrcQuad, ret->img.data,
                    ret->img.width, oDstROI, aDstQuad, interpolation CONTEXT));
        } else {
                assert(in->comp_count == 3);
                CHECK_NPP(NPP_CONTEXTIZE(nppiWarpPerspectiveQuad_8u_C3R)(
                    in->data, oSrcSize, 3 * in->width, oSrcROI, aSrcQuad,
                    ret->img.data, 3 * ret->img.width, oDstROI, aDstQuad,
                    interpolation CONTEXT));
        }
        GPU_TIMER_STOP(rotate);

        return ret;
}
