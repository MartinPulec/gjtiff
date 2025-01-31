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

struct rotate_state {
        cudaStream_t stream;

        uint8_t *output;
        size_t output_allocated;
};

struct rotate_state *rotate_init(cudaStream_t stream)
{
        struct rotate_state *s = calloc(1, sizeof *s);
        assert(s != NULL);
        s->stream = stream;

        return s;
}

void rotate_destroy(struct rotate_state *s)
{
        if (s == NULL) {
                return;
        }
        CHECK_CUDA(cudaFreeAsync(s->output, s->stream));
        free(s);
}

/**
 * normalize the coordinates to 0..1 and
 * @return asoect ratio
 */
static double normalize_coords(struct coordinate coords[4])
{
        double lat_min = 1e6;
        double lat_max = -1e6;
        double lon_min = 1e6;
        double lon_max = -1e6;

        enum {
                LAT_OFF = 360,
                LON_OFF = 180,
        };

        for (unsigned i = 0; i < 4; ++i) {
                // make both positive:
                // - lat - 0-180
                // - lon - 0-360
                coords[i].latitude += LAT_OFF;
                coords[i].longitude += LON_OFF;

                if (coords[i].latitude < lat_min) {
                        lat_min = coords[i].latitude;
                }
                if (coords[i].latitude > lat_max) {
                        lat_max = coords[i].latitude;
                }
                if (coords[i].longitude < lon_min) {
                        lon_min = coords[i].longitude;
                }
                if (coords[i].longitude > lon_max) {
                        lon_max = coords[i].longitude;
                }
        }

        // handle longitude 179->-179 transition (eastern to western
        // hemishpere); zero is now shifted to 180
        if (lon_min < LON_OFF && lon_max >= LON_OFF) {
                for (unsigned i = 0; i < 4; ++i) {
                        if (coords[i].longitude < LON_OFF) {
                                coords[i].longitude += 360.0;
                        }
                }
                return normalize_coords(coords);
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

/// fullfill GPUJPEG mem requirements
static void adjust_size(int *width, int *height, int comp_count) {

        enum {
                GB1 = 1ULL * 1000 * 1000 * 1000,
                THRESH_GB = 7ULL * GB1,
                GJ_PER_BYTE_REQ = 20,
        };
        size_t gj_gram_needed = (size_t)(*width * *height * comp_count) *
                                GJ_PER_BYTE_REQ;
        if (gj_gram_needed < THRESH_GB) {
                return;
        }
        WARN_MSG(
            "[rotate] Encoding of %dx%d image would require %.2f GB GRAM (>=%g "
            "GB), downsizing ",
            *width, *height, (double)gj_gram_needed / GB1,
            (double)THRESH_GB / GB1);
        while (gj_gram_needed > THRESH_GB) {
                *width /= 2;
                *height /= 2;
                gj_gram_needed /= 4;
        }
        WARN_MSG("rotated to %.2f GB (%dx%d).\n", (double)gj_gram_needed / GB1,
                 *width, *height);
}

struct dec_image rotate(struct rotate_state *s, const struct dec_image *in)
{
        if (s == NULL) {
                return *in;
        }
        if (!in->coords_set) {
                WARN_MSG("Coordinates not set, not normalizing image...\n");
                return *in;
        }

        double aSrcQuad[4][2] = {
            {0.0, 0.0},              // Top-left
            {in->width, 0},          // Top-right
            {in->width, in->height}, // Bottom-right
            {0.0, in->height}        // Bottom-left
        };

        struct coordinate coords[4];
        memcpy(coords, in->coords, sizeof coords);
        double dst_aspect = normalize_coords(coords);

        NppiRect oSrcROI = {0, 0, in->width, in->height};
        NppiSize oSrcSize = {in->width, in->height};

        struct dec_image ret = *in;
        if (dst_aspect >= 1) {
                ret.width = (int)(ret.height * dst_aspect);
        } else {
                ret.height = (int)(ret.width / dst_aspect);
        }
        adjust_size(&ret.width, &ret.height, in->comp_count);

        const size_t req_size = (size_t)ret.width * ret.height * ret.comp_count;
        if (req_size >= s->output_allocated) {
                CHECK_CUDA(cudaFree(s->output));
                s->output = NULL;
                CHECK_CUDA(cudaMalloc((void **)&s->output, req_size));
                s->output_allocated = req_size;
        }
        ret.data = s->output;

        NppiRect oDstROI = {0, 0, ret.width, ret.height};
        double aDstQuad[4][2] = {
            {coords[0].longitude * ret.width,
             coords[0].latitude * ret.height}, // Top-left
            {coords[1].longitude * ret.width,
             coords[1].latitude * ret.height}, // Top-right
            {coords[2].longitude * ret.width,
             coords[2].latitude * ret.height}, // Bottom-right
            {coords[3].longitude * ret.width,
             coords[3].latitude * ret.height}, // Bottom-left
        };

        const int interpolation = NPPI_INTER_LINEAR;
        if (in->comp_count == 1) {
                CHECK_NPP(nppiWarpPerspectiveQuad_8u_C1R(
                    in->data, oSrcSize, in->width, oSrcROI, aSrcQuad, ret.data,
                    ret.width, oDstROI, aDstQuad, interpolation));
        } else {
                assert(in->comp_count == 3);
                CHECK_NPP(nppiWarpPerspectiveQuad_8u_C3R(
                    in->data, oSrcSize, 3 * in->width, oSrcROI, aSrcQuad,
                    ret.data, 3 * ret.width, oDstROI, aDstQuad,
                    interpolation));
        }

        return ret;
}
