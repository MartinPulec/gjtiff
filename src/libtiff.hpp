#ifndef LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642
#define LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642

#include "defs.h"

struct libtiff_state;

struct libtiff_state *libtiff_create(int log_level, void *stream);
struct dec_image libtiff_decode(struct libtiff_state *s, const char *fname);
void libtiff_destroy(struct libtiff_state *s);

#endif // !defiend LIBTIFF_HPP_34DACD19_5FC8_4794_9338_EE857B47C642
