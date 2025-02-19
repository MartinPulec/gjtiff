#include <gdal.h>
#include <ogr_srs_api.h>
#include <stdio.h>
#include <stdlib.h>

void transform_and_print(double x, double y,
                         OGRCoordinateTransformationH transform,
                         const char *label)
{
        double x_copy = x;
        double y_copy = y;
        if (OCTTransform(transform, 1, &x_copy, &y_copy, NULL)) {
                printf("%-12s (%.3f, %.3f) -> (%.6f°, %.6f°)\n", label, x, y,
                       x_copy, y_copy);
        } else {
                fprintf(stderr, "Failed to transform coordinates for %s.\n",
                        label);
        }
}

int main(int argc, char *argv[])
{
        if (argc != 2) {
                printf("Usage: %s <input.jp2>\n", argv[0]);
                return 1;
        }

        const char *filename = argv[1];
        GDALAllRegister();

        GDALDatasetH dataset = GDALOpen(filename, GA_ReadOnly);
        if (dataset == NULL) {
                fprintf(stderr, "Failed to open file: %s\n", filename);
                return 1;
        }

        double geoTransform[6];
        if (GDALGetGeoTransform(dataset, geoTransform) != CE_None) {
                fprintf(stderr, "Failed to get geotransform.\n");
                GDALClose(dataset);
                return 1;
        }

        // Calculate corner coordinates
        double x_min = geoTransform[0];
        double y_max = geoTransform[3];
        double x_max = x_min + geoTransform[1] * GDALGetRasterXSize(dataset);
        double y_min = y_max + geoTransform[5] * GDALGetRasterYSize(dataset);
        double x_center = (x_min + x_max) / 2.0;
        double y_center = (y_min + y_max) / 2.0;

        // Get projection
        const char *proj_wkt = GDALGetProjectionRef(dataset);
        if (!proj_wkt || !proj_wkt[0]) {
                fprintf(stderr, "Dataset has no projection.\n");
                GDALClose(dataset);
                return 1;
        }

        OGRSpatialReferenceH src_srs = OSRNewSpatialReference(proj_wkt);
        OGRSpatialReferenceH dst_srs = OSRNewSpatialReference(NULL);
        OSRSetWellKnownGeogCS(dst_srs, "WGS84");

        OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(
            src_srs, dst_srs);
        if (!transform) {
                fprintf(stderr,
                        "Failed to create coordinate transformation.\n");
                OSRDestroySpatialReference(src_srs);
                OSRDestroySpatialReference(dst_srs);
                GDALClose(dataset);
                return 1;
        }

        // Print coordinates with transformations
        printf("Corner Coordinates (Projected -> Geographic):\n");
        transform_and_print(x_min, y_max, transform, "Upper Left");
        transform_and_print(x_min, y_min, transform, "Lower Left");
        transform_and_print(x_max, y_max, transform, "Upper Right");
        transform_and_print(x_max, y_min, transform, "Lower Right");
        transform_and_print(x_center, y_center, transform, "Center");

        // Cleanup
        OCTDestroyCoordinateTransformation(transform);
        OSRDestroySpatialReference(src_srs);
        OSRDestroySpatialReference(dst_srs);
        GDALClose(dataset);
        return 0;
}
