#include <cstdint>
#include <memory>
#include <cstdlib>   // for free

#include "defs.h"

struct libtiff_state {
        libtiff_state(int l);
        int log_level;
        std::unique_ptr<uint8_t[], void (*)(void *)> tmp_buffer{nullptr, free};
        size_t tmp_buffer_allocated = 0;
        std::unique_ptr<uint8_t[], void (*)(void *)> decoded{nullptr, free};
        size_t decoded_allocated{};
        struct dec_image decode(const char *fname, void *stream);
};
