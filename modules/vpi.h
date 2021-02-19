#ifndef VPI_H
#define VPI_H

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/ImageResampler.h>
#include <vpi/algo/ImageFormatConverter.h>

#define CHECK_STATUS(STMT)                                      \
    do                                                          \
    {                                                           \
        VPIStatus status = (STMT);                              \
        if (status != VPI_SUCCESS)                              \
        {                                                       \
            throw std::runtime_error(vpiStatusGetName(status)); \
        }                                                       \
    } while (0);

// Convert a cv::Mat to a VPI image
VPIImage create_vpi_image_from_mat(cv::Mat cv_image);

// Convert a VPI Image to a cv::Mat
cv::Mat create_mat_from_vpi_image(VPIImage vpi_image);

// Resize image
cv::Mat vpi_resize_image(cv::Mat cv_image, int height, int width, VPIDeviceType device_type);

// Convert image format
cv::Mat vpi_convert_image_format(cv::Mat cv_image, VPIDeviceType device_type);

#endif // VPI_H
