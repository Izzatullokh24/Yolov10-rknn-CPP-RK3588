#ifndef RKYOLOV11_H
#define RKYOLOV11_H

#include <string>
#include <mutex>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"

// Include the actual definitions instead of forward declarations
#include "common.h"
#include "image_utils.h"
#include "yolo11.h"
#include "postprocess.h"

// The adapter class - declarations only
class rkYolov11
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
    rkYolov11(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &orig_img);
    ~rkYolov11();
};

#endif // RKYOLOV11_H