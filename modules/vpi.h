#ifndef VPI_H
#define VPI_H

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif
  
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Types.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/ConvertImageFormat.h>

#define CHECK_STATUS(STMT)                                     \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);

// Convert a cv::Mat to a VPI image
VPIImage create_vpi_image_from_mat(cv::Mat cv_image);

// Convert a VPI Image to a cv::Mat
//cv::Mat create_mat_from_vpi_image(VPIImage vpi_image);

// Resize image
cv::Mat vpi_resize_image(cv::Mat cv_image, uint32_t height, uint32_t width, VPIBackend backend_type = VPIBackend::VPI_BACKEND_VIC);

// Convert image format
cv::Mat vpi_convert_image_format(cv::Mat cv_image, int cv_color_conversion_code, VPIBackend backend_type = VPIBackend::VPI_BACKEND_VIC);

#endif // VPI_H
