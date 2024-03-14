#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <nvtiff.h>
#include <memory>

struct libtiff_state {
  std::unique_ptr<uint8_t[], void (*)(void *)> tmp_buffer{nullptr, free};
  size_t tmp_buffer_allocated = 0;
  uint8_t *decode(const char *fname, size_t *nvtiff_out_size,
                  nvtiffImageInfo_t *image_info, uint8_t **decoded,
                  size_t *decoded_allocated, cudaStream_t stream);
};
