#include "png.h"

#include <cassert>
#include <cstdlib>    // for getenv
#include <cstring>
#include <sys/stat.h>

#include "defs.h"
#include "fpng.h"
#include "fpnge.h"
// #include "pam.h"
#include "utils.h"   // for ERROR_MSG

bool backend_fpng;

void png_init() {
        const char *backend = getenv("PNG_BACKEND");
        if (backend == nullptr) {
                return;
        }
        assert(strcasecmp(backend, "fpng") == 0);
        fpng::fpng_init();
        backend_fpng = true;
}

/// encode with fpng backend
static bool encode_fpng(const struct dec_image *img,
                        unsigned long width_padding, const char *ofname)
{
        assert(img->comp_count == 4);
        if (!fpng::fpng_encode_image_to_file(ofname, img->data, img->width,
                                             img->height, 4,
                                             width_padding, /* flags */ 0)) {
                ERROR_MSG("[png] Encoding %s failed!", ofname);
                return false;
        }
        return true;
}

/// encode with fpnge backend - default, FASTER
static bool encode_fpnge(const struct dec_image *img,
                         unsigned long width_padding, const char *ofname)
{
        assert(img->comp_count == 2 || img->comp_count == 4);
        struct FPNGEOptions opts;
        FPNGEFillOptions(&opts, 1, FPNGE_CICP_NONE);
        char *out = (char *)malloc(
            FPNGEOutputAllocSize(1, img->comp_count, img->width, img->height));
        FILE *outf = fopen(ofname, "wb");
        assert(outf != nullptr);
        size_t bytes = FPNGEEncode(1, img->comp_count, img->data, img->width,
                                   ((size_t)img->width * img->comp_count) +
                                       width_padding,
                                   img->height, out, &opts);
        int ret = fwrite(out, bytes, 1, outf);
        assert(ret != 0);
        free(out);
        fclose(outf);
        return true;
}

unsigned long encode_png(const struct dec_image *img,
                         unsigned long width_padding, const char *ofname) {
        // return pam_write(ofname, img->width, img->width + width_padding,
        //                  img->height, 4, 255, img->data, false)
        //            ? (size_t)img->width * img->height * 4
        //            : 0;
        bool ret = backend_fpng ? encode_fpng(img, width_padding, ofname)
                                : encode_fpnge(img, width_padding, ofname);
        if (!ret) {
                return 0;
        }

        struct stat file_info;
        if (stat(ofname, &file_info) == 0) {
                return file_info.st_size;
        }
        // weird if stat fails here but we can return uncompressed size
        return (size_t)img->width * img->height * img->comp_count;
}
