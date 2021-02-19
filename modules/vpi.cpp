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

// includes for OpenCV >= 3.x
#ifndef CV_VERSION_EPOCH
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

// OpenCV includes for OpenCV 2.x
#ifdef CV_VERSION_EPOCH
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#endif

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/ImageResampler.h>
#include <vpi/algo/ImageFormatConverter.h>

#include <cstring> // for memset

#include "vpi.h"

/**
 * Create a VPIImage object from a cv::Mat
 */
VPIImage create_vpi_image_from_mat(cv::Mat cv_image)
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
}

/**
 * Create a cv::Mat from a VPIImage object
 */
cv::Mat create_mat_from_vpi_image(VPIImage vpi_image)
{
    VPIImageData img_data;

    // Lock the image for safe access and create the Mat
    CHECK_STATUS(vpiImageLock(vpi_image, VPI_LOCK_READ, &img_data));
    cv::Mat cv_image(img_data.planes[0].height, img_data.planes[0].width, CV_8UC1, img_data.planes[0].data, img_data.planes[0].rowStride);
    CHECK_STATUS(vpiImageUnlock(vpi_image));

    return cv_image;
}

/**
 * Resize an image
 */
cv::Mat vpi_resize_image(cv::Mat cv_image, int height, int width, VPIDeviceType device_type)
{
    VPIImage input = create_vpi_image_from_mat(cv_image);
    VPIImage output;
    VPIStream stream;

    // Create a stream for the chosen backend
    CHECK_STATUS(vpiStreamCreate(device_type, &stream));

    // Output image container in the desired new size
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_TYPE_BGR8, 0, &output));

    // Resize the image
    CHECK_STATUS(vpiSubmitImageResampler(stream, input, output, VPI_INTERP_LINEAR_FAST, VPI_BOUNDARY_COND_ZERO));
    CHECK_STATUS(vpiStreamSync(stream));

    // Retrieve the output image
    cv::Mat result = create_mat_from_vpi_image(output);

    // Cleanup VPI resources
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    vpiStreamDestroy(stream);

    return result;
}

/**
 * Convert a cv::Mat from RGB8 to BGR8 format using the backend specified
 */
cv::Mat vpi_convert_image_format(cv::Mat cv_image, VPIDeviceType device_type)
{   
    VPIImage input = create_vpi_image_from_mat(cv_image);
    VPIImage output;
    VPIStream stream;

    // Create a stream for the chosen backend
    CHECK_STATUS(vpiStreamCreate(device_type, &stream));

    // Output image container in BGR8 format
    CHECK_STATUS(vpiImageCreate(cv_image.cols, cv_image.rows, VPI_IMAGE_TYPE_BGR8, 0, &output));

    // Convert the image
    CHECK_STATUS(vpiSubmitImageFormatConverter(stream, input, output, VPI_CONVERSION_CAST, 1, 0));
    CHECK_STATUS(vpiStreamSync(stream));

    // Retrieve the output image
    cv::Mat result = create_mat_from_vpi_image(output);

    // Cleanup VPI resources
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    vpiStreamDestroy(stream);

    return result;
}
