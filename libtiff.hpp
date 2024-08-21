#ifndef LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642
#define LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642

#include <cstdint>

#include "defs.h"

struct libtiff_state {
        libtiff_state(int l);
        ~libtiff_state();
        int log_level;

        uint8_t *decoded = nullptr;
        uint8_t *d_decoded = nullptr;
        size_t decoded_allocated{};

        uint8_t *d_converted = nullptr;
        size_t d_converted_allocated = 0;

        struct dec_image decode(const char *fname, void *stream);
};

#endif // !defiend LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642
