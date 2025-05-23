#ifndef RKYOLOV10_H
#define RKYOLOV10_H

#include <string>
#include <mutex>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"

// Include the actual definitions instead of forward declarations
#include "common.h"
#include "image_utils.h"
#include "yolov10.h"
#include "postprocess.h"

// The adapter class - declarations only
class rkYolov10
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    rknn_app_context_t* app_ctx;  // Use pointer instead of object
    float nms_threshold, box_conf_threshold;

    // Helper function to convert OpenCV Mat to image_buffer_t
    image_buffer_t convert_mat_to_image_buffer(cv::Mat &img);

public:
    rkYolov10(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &orig_img);
    ~rkYolov10();
};

#endif // RKYOLOV10_H
