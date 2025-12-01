#include "nd_processing.cuh"

#include "cuda_common.cuh"    // for bilinear_sample
#include "utils.h"

static __global__ void nd_process(struct dec_image out,
                                      struct dec_image in1,
                                      struct dec_image in2)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width|| out_y >= out.height) {
                return;
        }

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        float abs_pos_src_x = rel_pos_src_x * in1.width;
        float abs_pos_src_y = rel_pos_src_y * in1.height;
        float val1 = bilinearSample(
            in1.data, in1.width, 1, in1.height, abs_pos_src_x, abs_pos_src_y);
        abs_pos_src_x = rel_pos_src_x * in2.width;
        abs_pos_src_y = rel_pos_src_y * in2.height;
        float val2 = bilinearSample(
            in2.data, in2.width, 1, in2.height, abs_pos_src_x, abs_pos_src_y);

        val1 /= 255.0;
        val2 /= 255.0;
#ifdef GAMMA
        val1 = pow(val1, 1.0F / GAMMA);
        val2 = pow(val2, 1.0F / GAMMA);
#endif

        float res = (val1 - val2) / (val1 / val2);
#ifdef GAMMA
        res = pow(res, GAMMA);
#endif
        float res255 = 255 * __saturatef((res + 1.F) / 2.F);

        out.data[3 * (out_x + out_y * out.width)] = res255;
        out.data[3 * (out_x + out_y * out.width) + 1] = res255;
        out.data[3 * (out_x + out_y * out.width) + 2] = res255;
}

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream)
{
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        nd_process<<<grid, block, 0, stream>>>(*out, *in1, *in2);
        CHECK_CUDA(cudaGetLastError());
}
