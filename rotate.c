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
static double normalize_coords(struct coordinates coords[4])
{
        double lat_min = 90.0;
        double lat_max = -90.0;
        double lon_min = 180.0;
        double lon_max = -180.0;

        for (unsigned i = 0; i < 4; ++i) {
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

        double lat_range = lat_max - lat_min;
        double lon_range = lon_max - lon_min;

        for (unsigned i = 0; i < 4; ++i) {
                coords[i].latitude = 1.0 -
                                     (coords[i].latitude - lat_min) / lat_range;
                coords[i].longitude = (coords[i].longitude - lon_min) /
                                      lon_range;
        }

        double lat_mean = lat_min + ((lat_max - lat_min) / 2);
        double lon_lat_ratio = cos(M_PI * lat_mean / 180.0);

        return (lon_range / lat_range) * lon_lat_ratio;
}

struct dec_image rotate(struct rotate_state *s, const struct dec_image *in)
{
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

        struct coordinates coords[4];
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

        CHECK_NPP(nppiWarpPerspectiveQuad_8u_C1R(
            in->data, oSrcSize, in->width, oSrcROI, aSrcQuad, ret.data,
            ret.width, oDstROI, aDstQuad, NPPI_INTER_LINEAR));

        return ret;
}
