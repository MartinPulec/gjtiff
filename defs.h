#ifndef DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
#define DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

#include "utils.hpp" // ERROR_MSG

#ifdef __cplusplus
#include <cstdlib>   // EXIT_FAILURE
#else
#include <stdlib.h>  // EXIT_FAILURE
#endif

enum rc {
        ERR_NVCOMP_NOT_FOUND = -3,
        ERR_GENERIC = 0,
        SUCCESS = 1,
};

struct dec_image {
        enum rc rc;
        int width;
        int height;
        int comp_count;
        unsigned char *data;
};

#define CHECK_CUDA(call)                                                       \
        {                                                                      \
                cudaError_t err = call;                                        \
                if (cudaSuccess != err) {                                      \
                        ERROR_MSG(                                             \
                            "Cuda error in file '%s' in line %i : %s.\n",      \
                            __FILE__, __LINE__, cudaGetErrorString(err));      \
                        exit(EXIT_FAILURE);                                    \
                }                                                              \
        }

#endif // ! defined DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
