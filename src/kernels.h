#include <cuda_runtime.h>

struct dec_image;

#ifdef __cplusplus
#include <cinttypes>
#include <cstddef>
#else
#include <inttypes.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void convert_16_8_normalize_cuda(struct dec_image *in, uint8_t *out,
                                 cudaStream_t stream);
void normalize_8(struct dec_image *in, uint8_t *out, cudaStream_t stream);

void convert_complex_int_to_uint16(const int16_t *in, uint16_t *out, size_t count,
                         cudaStream_t stream);

void convert_rgba_grayscale(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream);
void convert_rgba_rgb(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream);
void convert_remove_pitch(uint8_t *in, uint8_t *out, int width, int spitch,
                          int height, void *stream);
void convert_remove_pitch_16(uint16_t *in, uint16_t *out, int width, int spitch,
                             int height, void *stream);
void downscale_image_cuda(const uint8_t *in, uint8_t *out, int comp_count,
                          int src_width, int src_height, int factor,
                          void *stream);
void combine_images_cuda(struct dec_image *out, const struct dec_image *in1,
                    const struct dec_image *in2, const struct dec_image *in3,
                    cudaStream_t stream);
[[nodiscard]] uint8_t *convert_rgb_to_yuv420(const struct dec_image *in,
                                             cudaStream_t stream);
[[nodiscard]] uint8_t *convert_y_full_to_limited(const struct dec_image *in,
                                                 cudaStream_t stream);

void rotate_set_alpha(struct dec_image *in, double aDstQuad[4][2],
                      cudaStream_t stream);

void cleanup_cuda_kernels(void);

void thrust_process_s2_band(uint16_t *d_ptr, uint8_t *d_alpha, size_t count,
                            cudaStream_t stream);
void thrust_16b_to_8b(uint16_t *d_in, uint8_t *d_out, size_t count,
                      cudaStream_t stream);

#ifdef __cplusplus
}
#endif
