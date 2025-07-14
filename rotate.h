#ifndef ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5
#define ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

#ifdef __cplusplus
extern "C" {
#endif

struct rotate_state;

struct rotate_state *rotate_init(cudaStream_t stream);
struct owned_image *rotate(struct rotate_state *state, const struct dec_image *in);
void rotate_destroy(struct rotate_state *s);


// util for print_bbox
void get_lat_lon_min_max(const struct coordinate coords[4], double *lat_min,
                         double *lat_max, double *lon_min, double *lon_max);

void gcs_to_wgs(const struct coordinate src_coords[4],
                struct coordinate coords[4]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5
