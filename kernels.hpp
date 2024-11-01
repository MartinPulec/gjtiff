#include <cinttypes>
#include <cstddef>
#include <cuda_runtime.h>

struct dec_image;

void convert_16_8_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream);

void convert_complex_int_to_uint16(const int16_t *in, uint16_t *out, size_t count,
                         cudaStream_t stream);

void convert_rgba_grayscale(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream);
void convert_rgba_rgb(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream);
void convert_planar_rgb_to_packed(uint8_t **in, uint8_t *out, int width,
                                  int spitch, int height, void *stream);

void cleanup_cuda_kernels(void);
