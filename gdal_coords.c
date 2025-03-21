#include <gdal.h>
#include <ogr_srs_api.h>
#include <pthread.h>
#include <stdlib.h>

#include "defs.h"
#include "utils.h"

static bool transform_and_print(double x, double y,
                                OGRCoordinateTransformationH transform,
                                struct coordinate *coord, const char *label)
{
        if (!OCTTransform(transform, 1, &x, &y, NULL)) {
                WARN_MSG("Failed to transform coordinates for %s.\n", label);
                return false;
        }
        coord->latitude = x;
        coord->longitude = y;
        INFO_MSG("\t%-11s: %f, %f\n", label, coord->latitude, coord->longitude);
        return true;
}

void set_coords_from_gdal(const char *fname, struct dec_image *image)
{
        static pthread_once_t gdal_init = PTHREAD_ONCE_INIT;
        pthread_once(&gdal_init, GDALAllRegister);

        GDALDatasetH dataset = GDALOpen(fname, GA_ReadOnly);
        if (dataset == NULL) {
                WARN_MSG("[gdal] Failed to open file: %s\n", fname);
                return;
        }

        double geoTransform[6];
        if (GDALGetGeoTransform(dataset, geoTransform) != CE_None) {
                WARN_MSG("[gdal] Failed to get geotransform (%s).\n", fname);
                GDALClose(dataset);
                return;
        }

        // Calculate corner coordinates
        double x_min = geoTransform[0];
        double y_max = geoTransform[3];
        double x_max = x_min + geoTransform[1] * GDALGetRasterXSize(dataset);
        double y_min = y_max + geoTransform[5] * GDALGetRasterYSize(dataset);

        // Get projection
        const char *proj_wkt = GDALGetProjectionRef(dataset);
        if (!proj_wkt || !proj_wkt[0]) {
                WARN_MSG("Dataset has no projection (%s).\n", fname);
                GDALClose(dataset);
                return;
        }

        OGRSpatialReferenceH src_srs = OSRNewSpatialReference(proj_wkt);
        OGRSpatialReferenceH dst_srs = OSRNewSpatialReference(NULL);
        OSRSetWellKnownGeogCS(dst_srs, "WGS84");

        OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(
            src_srs, dst_srs);
        if (!transform) {
                WARN_MSG("Failed to create coordinate transformation (%s).\n",
                         fname);
                OSRDestroySpatialReference(src_srs);
                OSRDestroySpatialReference(dst_srs);
                GDALClose(dataset);
                return;
        }

        // Print coordinates with transformations
        INFO_MSG("Got points:\n");
        bool success = true;
        success = success && // Upper Left
                  transform_and_print(x_min, y_max, transform,
                                      &image->coords[0], coord_pos_name[0]);
        success = success && // Upper Right
                  transform_and_print(x_max, y_max, transform,
                                      &image->coords[1], coord_pos_name[1]);
        success = success && // Lower Right
                  transform_and_print(x_max, y_min, transform,
                                      &image->coords[2], coord_pos_name[2]);
        success = success && // Lower Left
                  transform_and_print(x_min, y_min, transform,
                                      &image->coords[3], coord_pos_name[3]);
        image->coords_set = success;

        // Cleanup
        OCTDestroyCoordinateTransformation(transform);
        OSRDestroySpatialReference(src_srs);
        OSRDestroySpatialReference(dst_srs);
        GDALClose(dataset);
}
