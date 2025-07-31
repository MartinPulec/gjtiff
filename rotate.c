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
#include "rotate_utm.h"
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
        struct rotate_utm_state *rotate_utm;
};

struct rotate_state *rotate_init(cudaStream_t stream)
{
        struct rotate_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

#ifdef NPP_NEW_API
        init_npp_context(&s->nppStreamCtx, stream);
#endif

        s->rotate_utm = rotate_utm_init(stream);
        assert(s->rotate_utm != NULL);

        return s;
}

void rotate_destroy(struct rotate_state *s)
{
        if (s == NULL) {
                return;
        }
        rotate_utm_destroy(s->rotate_utm);
        free(s);
}

void get_lat_lon_min_max(const struct coordinate coords[4], double *lat_min,
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
static double normalize_coords(const struct coordinate src_coords[4],
                               struct coordinate coords[4], double bounds[4])
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
                gcs_to_webm(src_coords[i].latitude, src_coords[i].longitude,
                            &coords[i].latitude, &coords[i].longitude);
        }

        get_lat_lon_min_max(coords, &lat_min, &lat_max, &lon_min, &lon_max);
        bounds[XLEFT] = lon_min;
        bounds[YTOP] = lat_max;
        bounds[XRIGHT] = lon_max;
        bounds[YBOTTOM] = lat_min;

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

void adjust_size(int *width, int *height, int comp_count) {
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

        if (strncmp(in->authority, "ESPG:", strlen("ESPG:")) != 0) {
                struct owned_image *ret = rotate_utm(s->rotate_utm, in);
                if (ret != NULL) {
                        return ret;
                }
                WARN_MSG("rotate_utm returned nullptr!\n");
        } else if (strlen(in->authority) > 0) {
                WARN_MSG("Unsupported authority: %s!\n", in->authority);
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
        double bounds[4];
        const double dst_aspect = normalize_coords(in->coords, coords, bounds);
        if (dst_aspect == -1) {
                DEBUG_MSG("Near North/South pole - not rotating\n");
                return take_ownership(in);
        }
        assert(dst_aspect > 0);

        NppiRect oSrcROI = {0, 0, in->width, in->height};
        NppiSize oSrcSize = {in->width, in->height};

        // keep one side as in original and upscale the other to meet dst
        // projection dimension
        const double src_aspect = (double)in->width / in->height;
        struct dec_image dst_desc = *in;
        if (dst_aspect >= src_aspect) {
                dst_desc.width = (int)(dst_desc.height * dst_aspect);
        } else {
                dst_desc.height = (int)(dst_desc.width / dst_aspect);
        }
        adjust_size(&dst_desc.width, &dst_desc.height, in->comp_count);

        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        snprintf(ret->img.authority, sizeof ret->img.authority, "%s", "EPSG:4326");
        for (unsigned i = 0; i < ARR_SIZE(bounds); ++i) {
                ret->img.bounds[i] = bounds[i];
        }

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
