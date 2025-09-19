#include "rotate_tie_points.h"

#include <cassert>              // for assert
#include <cstdlib>              // for abort, calloc, free

// #include "cuda_common.cuh"
#include "defs.h"
#include "nppdefs.h"
// #include "rotate.h" // for get_lat_lon_min_max
#include "utils.h"              // for ERROR_MSG

struct rotate_tie_points_state {
        cudaStream_t stream;
};

struct rotate_tie_points_state *rotate_tie_points_init(cudaStream_t stream)
{
        struct rotate_tie_points_state *s = (struct rotate_tie_points_state *)calloc(
            1, sizeof *s);
        assert(s != nullptr);
        s->stream = stream;
        return s;
}

void rotate_tie_points_destroy(struct rotate_tie_points_state *s)
{
        if (s == nullptr) {
                return;
        }
        free(s);
}

struct owned_image *rotate_tie_points(struct rotate_tie_points_state *s, const struct dec_image *in)
{
        ERROR_MSG("noop\n");
        abort();
}
