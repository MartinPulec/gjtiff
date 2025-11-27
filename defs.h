#ifndef DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510
#define DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

#ifndef __cplusplus
#include <stdbool.h>
#include <stddef.h>  // for size_t
#else
#include <cstddef>  // for size_t
#endif

enum {
        LL_REALLY_QUIET = -2,
        LL_QUIET = -1,
        LL_INFO = 0,
        LL_VERBOSE = 1,
        LL_DEBUG = 2,
};
enum out_format { OUTF_NONE, OUTF_JPEG, OUTF_RAW, OUTF_WEBP };

// defined in main.c
extern int log_level;
extern size_t gpu_memory;
extern long long mem_limit;
extern bool alpha_wanted;
extern int fill_color;

enum rc {
        ERR_SOME_FILES_NOT_TRANSCODED = 2,
        ERR_NVCOMP_NOT_FOUND = 3,
};

struct coordinate {
        double latitude;
        double longitude;
};

enum coord_indices { ULEFT, URIGHT, BRIGHT, BLEFT, };
enum bound_indices { XLEFT, YTOP, XRIGHT, YBOTTOM, };

#define EARTH_PERIMETER 20037508.342789244

enum epsg {
        EPSG_WEB_MERCATOR = 3857, ///< coords normalized to 0..1 and vertical
                                  ///< axis top-bottom (0.0 == ~85 N)
        EPSG_WGS_84 = 4326,
        // North
        EPSG_UTM_1N = 32601,
        EPSG_UTM_60N = 32660,
        // South
        EPSG_UTM_1S = 32701,
        EPSG_UTM_60S = 32760,
};

struct tie_point {
        union {
                float lat;
                float webx;
        };
        union {
                float lon;
                float weby;
        };
        unsigned short x;
        unsigned short y;
};

struct tie_points {
        struct tie_point *points;
        unsigned count;
        unsigned grid_width; ///< number of tie points on line
};

struct dec_image {
        enum rc rc; ///< defined only if data=nullptr
        int width;
        int height;
        int comp_count;
        unsigned char *data;
        unsigned char *alpha;

        /// order: uppper left, upper right, lower right, lower left
        /// (@sa coord_pos_name)
        struct coordinate coords[4];
        bool coords_set;

        char authority[20]; // EPSG:xxxx or "" if not set
        double bounds[4]; // order - enum bound_indices

        int e3857_sug_w, e3857_sug_h; // ESPG:3857 suggested dimensions
        bool is_slc;

        struct tie_points tie_points;
};

struct owned_image {
        struct dec_image img;
        void (*free)(struct owned_image *img);
};

#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#if !defined __cplusplus && __STDC_VERSION__ <= 202311L
#define nullptr NULL
#endif

/// normalized difference feature (feat1-feat2)/(feat1+feat2)
enum nd_feature {
        ND_UNKNOWN,
        NDVI, ///< Normalized Difference Vegetation Index (B8, B4)
        NDMI, ///< Normalized Difference Moisture Index (B8A, B11)
        NDWI, ///< Normalized Difference Water Index (B3, B8)
        NDSI, ///< Normalised Difference Snow Index (B3, B11)
};

#endif // ! defined DEFS_H_56B475E2_92D1_4894_BD86_866CE6EE0510

