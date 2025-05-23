#include <stdio.h>
#include <mutex>
#include <string>
#include "rknn_api.h"

// Include all the necessary headers for implementation
#include "yolo11.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "rkYolov11.hpp"

// Implementation of the helper function
image_buffer_t rkYolov11::convert_mat_to_image_buffer(cv::Mat &img) {
    image_buffer_t image;
    image.width = img.cols;
    image.height = img.rows;
    image.width_stride = img.cols;
    image.height_stride = img.rows;
    image.format = IMAGE_FORMAT_RGB888;  // Assuming RGB format
    image.virt_addr = img.data;
    image.size = img.cols * img.rows * 3;  // 3 channels for RGB
    image.fd = -1;  // Not using file descriptor
    return image;
}

// Constructor implementation
rkYolov11::rkYolov11(const std::string &model_path) {
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;
    box_conf_threshold = BOX_THRESH;
    
    // Allocate app_ctx
    app_ctx = new rknn_app_context_t();
    memset(app_ctx, 0, sizeof(rknn_app_context_t));
}

// Initialize the model
int rkYolov11::init(rknn_context *ctx_in, bool isChild) {
    std::lock_guard<std::mutex> lock(mtx);

    // If this is a child instance and we have a parent context to duplicate
    if (isChild && ctx_in) {
        // Create a new context by duplicating the parent
        app_ctx->rknn_ctx = 0;
        ret = rknn_dup_context(ctx_in, &app_ctx->rknn_ctx);
        if (ret < 0) {
            printf("rknn_dup_context error ret=%d\n", ret);
            return -1;
        }
        
        // Get model info from the duplicated context
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
        if (ret < 0) {
            printf("rknn_query error ret=%d\n", ret);
            return -1;
        }
        
        // Allocate and query input attributes
        app_ctx->input_attrs = (rknn_tensor_attr *)calloc(app_ctx->io_num.n_input, sizeof(rknn_tensor_attr));
        for (int i = 0; i < app_ctx->io_num.n_input; i++) {
            app_ctx->input_attrs[i].index = i;
            ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_query error ret=%d\n", ret);
                return -1;
            }
        }
        
        // Allocate and query output attributes
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
        
        // Initialize post-processing
        init_post_process();
        
        return 0;
    } else {
        // Initialize the model directly for parent instance
        ret = init_yolo11_model(model_path.c_str(), app_ctx);
        if (ret < 0) {
            printf("init_yolo11_model error ret=%d\n", ret);
            return -1;
        }
        
        // Initialize post-processing
        init_post_process();
        
        return 0;
    }
}

// Get the context pointer
rknn_context *rkYolov11::get_pctx() {
    return &app_ctx->rknn_ctx;
}

// Perform inference
cv::Mat rkYolov11::infer(cv::Mat &orig_img) {
    std::lock_guard<std::mutex> lock(mtx);
    
    // Convert OpenCV Mat to BGR first (if not already in BGR format)
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
    ret = inference_yolo11_model(app_ctx, &image, &detect_result_list);
    if (ret < 0) {
        printf("inference_yolo11_model error ret=%d\n", ret);
        return orig_img; // Return original image on error
    }
    
    // Draw detection boxes on the original image
    for (int i = 0; i < detect_result_list.count; i++) {
        object_detect_result *det_result = &(detect_result_list.results[i]);
        
        // Get class name
        char *class_name = coco_cls_to_name(det_result->cls_id);
        
        // Format text with class name and confidence
        char text[256];
        sprintf(text, "%s %.1f%%", class_name, det_result->prop * 100);
        
        // Get bounding box coordinates
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        
        // Draw bounding box with green color for YOLOv11
        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        // Add text
        cv::putText(orig_img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    
    return orig_img;
}

// Destructor
rkYolov11::~rkYolov11() {
    // Release resources
    deinit_post_process();
    release_yolo11_model(app_ctx);
    
    // Free app_ctx
    delete app_ctx;
    app_ctx = nullptr;
}