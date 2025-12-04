#ifndef CUDA_COMMON_CUH_5CD7C953_A39A_4612_BBA3_4E47A0E7F659
#define CUDA_COMMON_CUH_5CD7C953_A39A_4612_BBA3_4E47A0E7F659

#include <cstdint>

// Device function: bilinear sample at (x, y) in [0..W) × [0..H)
template <typename T>
static __device__ __forceinline__ T bilinearSample(const T *src, int W,
                                                   int w_stride, int H, float x,
                                                   float y)
{
        // Compute integer bounds
        int x0 = int(floorf(x));
        int y0 = int(floorf(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // // Clamp to image edges - seem to happen at right
        x0 = max(0, min(x0, W - 1));
        y0 = max(0, min(y0, H - 1));
        x1 = max(0, min(x1, W - 1));
        y1 = max(0, min(y1, H - 1));

        // Fetch four neighbors
        float I00 = src[w_stride * (y0 * W + x0)];
        float I10 = src[w_stride * (y0 * W + x1)];
        float I01 = src[w_stride * (y1 * W + x0)];
        float I11 = src[w_stride * (y1 * W + x1)];

        // fractional part
        float dx = x - float(x0);
        float dy = y - float(y0);

        // interpolate in x direction
        float a = I00 + dx * (I10 - I00);
        float b = I01 + dx * (I11 - I01);

        // interpolate in y direction
        return a + dy * (b - a);
}

#endif // not defined CUDA_COMMON_CUH_5CD7C953_A39A_4612_BBA3_4E47A0E7F659
