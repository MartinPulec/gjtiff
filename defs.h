#ifndef DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
#define DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

#include "utils.hpp" // ERROR_MSG

#ifdef __cplusplus
#include <cstdlib>   // abort
#else
#include <stdlib.h>  // abort
#endif

enum rc {
        ERR_SOME_FILES_NOT_TRANSCODED = 2,
        ERR_NVCOMP_NOT_FOUND = 3,
};

struct dec_image {
        enum rc rc; ///< defined only if data=nullptr
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
                        abort();                                               \
                }                                                              \
        }
#endif // ! defined DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
