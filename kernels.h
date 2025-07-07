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

void convert_16_8_normalize_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream);
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

void convert_u8_to_float(const uint8_t *in, float *out, size_t count,
                         void *stream);
struct ramp;
uint32_t *get_lut(int ramp_sz, const struct ramp *ramp, void *stream);
void convert_float_to_u32(const float *d_in, uint32_t *d_out, size_t count,
                          const uint32_t *d_lut, void *stream);
void npp_float_operation(const float *in1, const float *in2,
                         float *out, char op, size_t count, void *stream);

void cleanup_cuda_kernels(void);

#ifdef __cplusplus
}
#endif
