#include <cassert>
#include <cstdio>
#include <tiffio.h>

#include "libtiff.hpp"
#include "kernels.hpp"
#include "utils.hpp"

using std::unique_ptr;

void nullTIFFErrorHandler(const char *, const char *, va_list) {}

libtiff_state::libtiff_state(int l) : log_level(l) {
  if (l == 0) {
    TIFFSetWarningHandler(nullTIFFErrorHandler);
  }
}

uint8_t *libtiff_state::decode(const char *fname, size_t *nvtiff_out_size,
                               nvtiffImageInfo_t *image_info, uint8_t **decoded,
                               size_t *decoded_allocated, cudaStream_t stream) {
  TIFF *tif = TIFFOpen(fname, "r");
  if (!tif) {
    fprintf(stderr, "libtiff cannot open image %s!\n", fname);
    return nullptr;
  }
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &image_info->image_width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &image_info->image_height);
  image_info->bits_per_sample[0] = 8;
  TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &image_info->samples_per_pixel);
  if (log_level > 1) {
    fprintf(stderr,
            "TIFF file %s: %" PRIu32 "x%" PRIu32 " %" PRIu32 " bits, %" PRIu32
            " sample(s)\n",
            fname, image_info->image_width, image_info->image_height,
            image_info->bits_per_sample[0], image_info->samples_per_pixel);
  }
  const size_t read_size =
      image_info->image_width * image_info->image_height * sizeof(uint32_t);
  if (read_size > tmp_buffer_allocated) {
    void *ptr = nullptr;
    cudaMallocManaged(&ptr, read_size);
    auto cfree = [](void *ptr) { cudaFree(ptr); };
    tmp_buffer = unique_ptr<uint8_t[], void (*)(void *)>((uint8_t *)ptr, cfree);
    assert(tmp_buffer.get() != nullptr);
    tmp_buffer_allocated = read_size;
  }
  TIMER_START(TIFFReadRGBAImage, log_level);
  /// @todo
  // TIFFReadRow{Tile,Strip} would be faster
  const int rc =
      TIFFReadRGBAImageOriented(tif, image_info->image_width, image_info->image_height,
                        (uint32_t *)tmp_buffer.get(), ORIENTATION_TOPLEFT, 0);
  TIMER_STOP(TIFFReadRGBAImage, log_level);
  TIFFClose(tif);
  if (rc != 1) {
    fprintf(stderr, "libtiff decode image %s failed!\n", fname);
    return nullptr;
  }
  const size_t out_size = image_info->image_width *
                             image_info->image_height *
                             image_info->samples_per_pixel;
  if (out_size > *decoded_allocated) {
    cudaFree(*decoded);
    cudaMalloc(decoded, out_size);
    *decoded_allocated = out_size;
  }
  switch (image_info->samples_per_pixel) {
  case 1:
    convert_rgba_grayscale(tmp_buffer.get(), *decoded,
                           image_info->image_width * image_info->image_height,
                           stream);
    break;
  case 3:
    convert_rgba_rgb(tmp_buffer.get(), *decoded,
                     image_info->image_width * image_info->image_height,
                     stream);
    break;
  default:
    fprintf(stderr, "Unsupported sample count %d!\n",
            image_info->samples_per_pixel);
    return nullptr;
  }
  *nvtiff_out_size = out_size;
  return *decoded;
}
