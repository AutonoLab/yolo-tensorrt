#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <atomic>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include <cstring> // for memset

#include "vpi.h"


/**
 * Create a VPIImage object from a cv::Mat
 */
/*VPIImage create_vpi_image_from_mat(cv::Mat cv_image)
{
    VPIImage image;

    // Containerize the cv::Mat in a VPIImageData struct
    VPIImageData img_data;
    memset(&img_data, 0, sizeof(img_data));
    img_data.type = VPI_IMAGE_TYPE_U8; // Corresponds with OpenCV type CV_8UC1
    img_data.numPlanes = 1;
    img_data.planes[0].width = cv_image.cols;
    img_data.planes[0].height = cv_image.rows;
    img_data.planes[0].rowStride = cv_image.step[0];
    img_data.planes[0].data = cv_image.data;

    CHECK_STATUS(vpiImageWrapHostMem(&img_data, 0, &image));

    return image;
}*/

/**
 * Create a cv::Mat from a VPIImage object
 */
cv::Mat create_mat_from_vpi_image(VPIImage vpi_image)
{
    VPIImageData img_data;

    // Lock the image for safe access and create the Mat
    CHECK_STATUS(vpiImageLock(vpi_image, VPI_LOCK_READ, &img_data));
    cv::Mat cv_image(img_data.planes[0].height, img_data.planes[0].width, CV_8UC1, img_data.planes[0].data, img_data.planes[0].pitchBytes);
    CHECK_STATUS(vpiImageUnlock(vpi_image));

    return cv_image;
}

/**
 * Resize an image
 * 
 * cv_image = input image that needs to be resized
 * height = height for new resized image
 * width = width for new resized image
 * backend_type = backend hardware to use for image resizing
 */
cv::Mat vpi_resize_image(cv::Mat cv_image, uint32_t height, uint32_t width, VPIBackend backend_type)
{
    VPIImage input;
    VPIImage output;
    VPIStream stream;

    // Create a VPI image from cv::Mat
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cv_image, 0, &input));

    // Create a stream for the chosen backend
    CHECK_STATUS(vpiStreamCreate(backend_type, &stream));

    // Output image container in the desired new size
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, 0, &output));

    // Resize the image
    CHECK_STATUS(vpiSubmitRescale(stream, backend_type, input, output, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0));
    CHECK_STATUS(vpiStreamSync(stream));

    VPIImageData img_data;

    // Lock the image for safe access and create the Mat
    CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &img_data));
    cv::Mat result(img_data.planes[0].height, img_data.planes[0].width, CV_8UC1, img_data.planes[0].data, img_data.planes[0].pitchBytes);
    CHECK_STATUS(vpiImageUnlock(output));

    // Ensure the stram is synchronized
    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }

    // Cleanup VPI resources
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    vpiStreamDestroy(stream);

    return result;
}

/**
 * Convert a cv::Mat from RGB8 to BGR8 format using the backend specified
 */
cv::Mat vpi_convert_image_format(cv::Mat cv_image, int cv_color_conversion_code, VPIBackend backend_type)
{   
    VPIImage input;
    VPIImage output;
    VPIStream stream;

    // Create a VPI image from cv::Mat
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cv_image, 0, &input));

    // Create a stream for the chosen backend
    CHECK_STATUS(vpiStreamCreate(backend_type, &stream));

    // Output image container in the desired new size
    CHECK_STATUS(vpiImageCreate(cv_image.cols, cv_image.rows, VPI_IMAGE_FORMAT_BGR8, 0, &output));

    // Convert the image
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backend_type, input, output, NULL));
    CHECK_STATUS(vpiStreamSync(stream));

    VPIImageData img_data;

    // Lock the image for safe access and create the Mat
    CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &img_data));
    cv::Mat result(img_data.planes[0].height, img_data.planes[0].width, CV_8UC1, img_data.planes[0].data, img_data.planes[0].pitchBytes);
    CHECK_STATUS(vpiImageUnlock(output));

    // Ensure the stram is synchronized
    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }

    // Cleanup VPI resources
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    vpiStreamDestroy(stream);

    return result;
}
