#include <stdio.h>
#include <string.h>
#include <mutex>
#include "rknn_api.h"

// Include all the necessary headers for implementation
#include "yolov10.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "rkYolov10.hpp"

// Implementation of the helper function
image_buffer_t rkYolov10::convert_mat_to_image_buffer(cv::Mat &img) {
    image_buffer_t image;
    image.width = img.cols;
    image.height = img.rows;
    image.width_stride = img.cols;
    image.height_stride = img.rows;
    image.format = IMAGE_FORMAT_RGB888;  
    image.virt_addr = img.data;
    image.size = img.cols * img.rows * 3;  
    image.fd = -1;  
    return image;
}

// Constructor implementation
rkYolov10::rkYolov10(const std::string &model_path) {
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;
    box_conf_threshold = BOX_THRESH;
    
    // Allocate app_ctx
    app_ctx = new rknn_app_context_t();
    memset(app_ctx, 0, sizeof(rknn_app_context_t));
}

// Initialize the model
int rkYolov10::init(rknn_context *ctx_in, bool isChild) {
    std::lock_guard<std::mutex> lock(mtx);

    if (isChild && ctx_in) {
        // Duplicate context for child instances
        app_ctx->rknn_ctx = 0;
        ret = rknn_dup_context(ctx_in, &app_ctx->rknn_ctx);
        if (ret < 0) {
            printf("rknn_dup_context error ret=%d\n", ret);
            return -1;
        }
        
        // Query model info from duplicated context
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
        if (ret < 0) {
            printf("rknn_query error ret=%d\n", ret);
            return -1;
        }
        
        // Set up input/output attributes
        app_ctx->input_attrs = (rknn_tensor_attr *)calloc(app_ctx->io_num.n_input, sizeof(rknn_tensor_attr));
        for (int i = 0; i < app_ctx->io_num.n_input; i++) {
            app_ctx->input_attrs[i].index = i;
            ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_query error ret=%d\n", ret);
                return -1;
            }
        }
        
        app_ctx->output_attrs = (rknn_tensor_attr *)calloc(app_ctx->io_num.n_output, sizeof(rknn_tensor_attr));
        for (int i = 0; i < app_ctx->io_num.n_output; i++) {
            app_ctx->output_attrs[i].index = i;
            ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_query error ret=%d\n", ret);
                return -1;
            }
        }
        
        // Set model dimensions
        if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
            app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
            app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
        } else {
            app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
            app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
            app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
        }
        
        // Set quantization flag
        if (app_ctx->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && 
            app_ctx->output_attrs[0].type == RKNN_TENSOR_UINT8) {
            app_ctx->is_quant = true;
        } else {
            app_ctx->is_quant = false;
        }
        
        init_post_process();
        return 0;
    } else {
        // Initialize model directly for parent instance
        ret = init_yolov10_model(model_path.c_str(), app_ctx);
        if (ret < 0) {
            printf("init_yolov10_model error ret=%d\n", ret);
            return -1;
        }
        
        init_post_process();
        return 0;
    }
}

// Get the context pointer
rknn_context *rkYolov10::get_pctx() {
    return &app_ctx->rknn_ctx;
}

// Perform inference
cv::Mat rkYolov10::infer(cv::Mat &orig_img) {
    std::lock_guard<std::mutex> lock(mtx);
    
    cv::Mat img;
    if (orig_img.channels() == 3 && orig_img.type() != CV_8UC3) {
        orig_img.convertTo(img, CV_8UC3);
    } else {
        img = orig_img.clone();
    }
    
    // Convert BGR to RGB for YOLO model
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // Convert Mat to image_buffer_t
    image_buffer_t image = convert_mat_to_image_buffer(img);
    
    // Prepare result container
    object_detect_result_list detect_result_list;
    memset(&detect_result_list, 0, sizeof(object_detect_result_list));
    
    // Run inference
    ret = inference_yolov10_model(app_ctx, &image, &detect_result_list);
    if (ret < 0) {
        printf("inference_yolov10_model error ret=%d\n", ret);
        return orig_img; 
    }
    
    // Draw detection boxes - use blue color for YOLOv10 to distinguish from v11
    for (int i = 0; i < detect_result_list.count; i++) {
        object_detect_result *det_result = &(detect_result_list.results[i]);
        
        char *class_name = coco_cls_to_name(det_result->cls_id);
        
        char text[256];
        sprintf(text, "%s %.1f%%", class_name, det_result->prop * 100);
        
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        
        // Use blue color for YOLOv10
        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
        cv::putText(orig_img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }
    
    return orig_img;
}

// Destructor
rkYolov10::~rkYolov10() {
    deinit_post_process();
    release_yolov10_model(app_ctx);
    
    delete app_ctx;
    app_ctx = nullptr;
}
