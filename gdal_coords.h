#ifndef GDAL_COORDS_H_BCECAC05_F647_4412_AB82_80B798E43BB5
#define GDAL_COORDS_H_BCECAC05_F647_4412_AB82_80B798E43BB5

#ifdef __cplusplus
extern "C" {
#endif

struct dec_image;
void set_coords_from_gdal(const char *fname, struct dec_image *image);

#ifdef __cplusplus
}
#endif

#endif // !defined GDAL_COORDS_H_BCECAC05_F647_4412_AB82_80B798E43BB5
