#include "png.h"

#include <cassert>
#include <sys/stat.h>

#include "defs.h"
#include "fpng.h"
// #include "pam.h"
#include "utils.h"   // for ERROR_MSG

void png_init() { fpng::fpng_init(); }

unsigned long encode_png(const struct dec_image *img,
                         unsigned long width_padding, const char *ofname)
{
        assert(img->comp_count == 4);
        // return pam_write(ofname, img->width, img->width + width_padding,
        //                  img->height, 4, 255, img->data, false)
        //            ? (size_t)img->width * img->height * 4
        //            : 0;
        if (!fpng::fpng_encode_image_to_file(ofname, img->data, img->width,
                                             img->height, 4,
                                             width_padding, /* flags */ 0)) {
                ERROR_MSG("[png] Encoding %s failed!", ofname);
                return 0;
        }
        struct stat file_info;
        if (stat(ofname, &file_info) == 0) {
                return file_info.st_size;
        }
        // weird if stat fails here but we can return uncompressed size
        return (size_t)img->width * img->height * 4;
}
