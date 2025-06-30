#ifndef DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
#define DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

#include "utils.h"   // ERROR_MSG

#ifndef __cplusplus
#include <stdbool.h>
#endif

enum {
        LL_REALLY_QUIET = -2,
        LL_QUIET = -1,
        LL_INFO = 0,
        LL_VERBOSE = 1,
        LL_DEBUG = 2,
};
extern int log_level;
extern size_t gpu_memory;

enum rc {
        ERR_SOME_FILES_NOT_TRANSCODED = 2,
        ERR_NVCOMP_NOT_FOUND = 3,
};

struct coordinate {
        double latitude;
        double longitude;
};

struct dec_image {
        enum rc rc; ///< defined only if data=nullptr
        int width;
        int height;
        int comp_count;
        unsigned char *data;

        /// order: uppper left, upper right, lower right, lower left
        /// (@sa coord_pos_name)
        struct coordinate coords[4];
        bool coords_set;
};

struct owned_image {
        struct dec_image img;
        void (*free)(struct owned_image *img);
};

#define DEC_IMG_ERR(rc) {rc, 0, 0, 0, 0, {0.0, 0.0}, false}

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
#define CHECK_NPP(call)                                                        \
        {                                                                      \
                NppStatus rc = call;                                           \
                if (NPP_NO_ERROR != rc) {                                      \
                        ERROR_MSG(                                             \
                            "NPP error in file '%s' in line %i : %s (%d).\n",  \
                            __FILE__, __LINE__, npp_status_to_str(rc),         \
                            (int)rc);                                          \
                        abort();                                               \
                }                                                              \
        }

#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif // ! defined DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

