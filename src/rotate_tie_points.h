#ifndef ROTATE_TIE_POINTS_H_f33a2775_7876_45bb_bf63_4355cf1874e2
#define ROTATE_TIE_POINTS_H_f33a2775_7876_45bb_bf63_4355cf1874e2

#include <cuda_runtime.h>  // for cudaStream_t

#ifdef __cplusplus
extern "C" {
#endif
 
struct dec_image;
struct rotate_tie_points_state;
struct owned_image;

struct rotate_tie_points_state *rotate_tie_points_init(cudaStream_t stream);
struct owned_image *rotate_tie_points(struct rotate_tie_points_state *state,
                                      const struct dec_image *in);
void rotate_tie_points_destroy(struct rotate_tie_points_state *s);

#ifdef __cplusplus
}
#endif

#endif // not defined ROTATE_TIE_POINTS_H_f33a2775_7876_45bb_bf63_4355cf1874e2
