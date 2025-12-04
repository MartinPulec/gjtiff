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
#include "kernels.h"
#include "nppdefs.h"
#include "nppi_geometry_transforms.h"
#include "rotate_tie_points.h"
#include "rotate_utm.h"
#include "utils.h"

struct rotate_state {
        bool disabled;
        cudaStream_t stream;
        NppStreamContext nppStreamCtx;
        struct rotate_utm_state *rotate_utm;
        struct rotate_tie_points_state *rotate_tp;
};

struct rotate_state *rotate_init(cudaStream_t stream, bool disabled)
{
        struct rotate_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->disabled = disabled;
        s->stream = stream;

#ifdef NPP_NEW_API
        init_npp_context(&s->nppStreamCtx, stream);
#endif

        s->rotate_tp = rotate_tie_points_init(stream);
        assert(s->rotate_tp != NULL);
        s->rotate_utm = rotate_utm_init(stream);
        assert(s->rotate_utm != NULL);

        return s;
}

void rotate_destroy(struct rotate_state *s)
{
        if (s == NULL) {
                return;
        }
        rotate_tie_points_destroy(s->rotate_tp);
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
static double normalize_coords(const struct coordinate src_coords[static 4],
                               struct coordinate coords[static 4],
                               double bounds[static 4])
{
        double lat_min = 0;
        double lat_max = 0;
        double lon_min = 0;
        double lon_max = 0;

        // check if not near poles
        get_lat_lon_min_max(src_coords, &lat_min, &lat_max, &lon_min, &lon_max);
        if (lat_min < -85. || lat_max > 85.) {
                [[maybe_unused]] const double near_pole_pt_lat =
                    (lat_max > 85 ? lat_max : lat_min);
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
        bounds[YTOP] = lat_min;
        bounds[XRIGHT] = lon_max;
        bounds[YBOTTOM] = lat_max;

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

static void release_owned_image(struct owned_image *img) {
        CHECK_CUDA(cudaFree(img->img.data));
        free(img);
}

static struct owned_image *take_ownership(const struct dec_image *in)
{
        struct owned_image *ret = malloc(sizeof *ret);
        memcpy(&ret->img, in, sizeof *in);
        const size_t size = (size_t) in->width * in->height * in->comp_count * 2;
        CHECK_CUDA(cudaMalloc((void **)&ret->img.data, size));
        CHECK_CUDA(
            cudaMemcpy(ret->img.data, in->data, size, cudaMemcpyDefault));
        ret->free = release_owned_image;
        return ret;
}

static bool is_utm(const char *authority) {
        if (strncmp(authority, "EPSG:", strlen("EPSG:")) != 0) {
                return false;
        }
        const int epsg_num = atoi(strchr(authority, ':') + 1);
        return (epsg_num >= EPSG_UTM_1N && epsg_num <= EPSG_UTM_60N) ||
               (epsg_num >= EPSG_UTM_1S && epsg_num <= EPSG_UTM_60S);
}

struct owned_image *rotate(struct rotate_state *s, const struct dec_image *in)
{
        assert(s != nullptr);
        if (!in->coords_set) {
                WARN_MSG("Coordinates not set, not normalizing image...\n");
                return take_ownership(in);
        }

        struct owned_image *ret = NULL;

        if (s->disabled) {
                ret = take_ownership(in);
                goto set_bounds; // avoid running eventual delegates
        }
        // delegates
        if (is_utm(in->authority)) {
                ret = rotate_utm(s->rotate_utm, in);
        } else if (in->tie_points.count > 0) {
                ret = rotate_tie_points(s->rotate_tp, in);
        }
        if (ret != NULL) {
                return ret;
        }

        // fallback follows
        if (strlen(in->authority) > 0) { // not UTM but defined...
                WARN_MSG("Unsupported authority: %s!\n", in->authority);
        }

set_bounds:;
        struct coordinate coords[4] = {};
        double bounds[4] = {};
        const double dst_aspect = normalize_coords(in->coords, coords, bounds);
        // rotation won't be performed
        if (dst_aspect == -1) {
                DEBUG_MSG("Near North/South pole - not rotating\n");
                ret = take_ownership(in);
        } else if (in->is_slc) {
                WARN_MSG("SLC product detected, not rotating...\n");
                ret = take_ownership(in);
        }
        // will not rotate but set at least the bounds (+authority)
        if (ret != NULL) {
                snprintf(ret->img.authority, sizeof ret->img.authority, "%s",
                         "EPSG:4326");
                for (unsigned i = 0; i < ARR_SIZE(bounds); ++i) {
                        ret->img.bounds[i] = bounds[i];
                }
                return ret;
        }
        assert(dst_aspect > 0);

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

        dst_desc.alpha = alpha_wanted ? (unsigned char *) 1 : NULL;
        ret = new_cuda_owned_image(&dst_desc);
        snprintf(ret->img.authority, sizeof ret->img.authority, "%s", "EPSG:4326");
        for (unsigned i = 0; i < ARR_SIZE(bounds); ++i) {
                ret->img.bounds[i] = bounds[i];
        }

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
        CHECK_CUDA(cudaMemsetAsync(ret->img.data, fill_color,
                                   (size_t)ret->img.width * ret->img.height *
                                       ret->img.comp_count,
                                   s->stream));
        const int interpolation = NPPI_INTER_LINEAR;

        NppStatus (*nppi_warp)(
            const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
            NppiRect oSrcROI, const double aSrcQuad[4][2], Npp8u *pDst,
            int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2],
            int eInterpolation, NppStreamContext nppStreamCtx) = NULL;

        if (in->comp_count == 1) {
                nppi_warp = nppiWarpPerspectiveQuad_8u_C1R_Ctx;
        } else {
                assert(in->comp_count == 3);
                nppi_warp = nppiWarpPerspectiveQuad_8u_C3R_Ctx;
        }
        CHECK_NPP(nppi_warp(in->data, oSrcSize, in->width, oSrcROI, aSrcQuad,
                            ret->img.data, ret->img.width, oDstROI, aDstQuad,
                            interpolation, s->nppStreamCtx));
        if (ret->img.alpha != NULL) {
                rotate_set_alpha(&ret->img, aDstQuad, s->stream);
        }
        GPU_TIMER_STOP(rotate);

        return ret;
}
