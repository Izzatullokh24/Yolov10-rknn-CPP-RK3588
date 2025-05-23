#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

// Global flag for clean termination
volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    printf("Caught signal %d, cleaning up and exiting...\n", sig);
    keep_running = 0;
}

// Function to process a frame with YOLO model
int process_frame(rknn_app_context_t *rknn_app_ctx, cv::Mat &frame) {
    int ret;
    object_detect_result_list od_results;
    image_buffer_t img_buffer;
    
    // Convert OpenCV Mat to image_buffer_t
    img_buffer.width = frame.cols;
    img_buffer.height = frame.rows;
    img_buffer.format = IMAGE_FORMAT_RGB888;
    img_buffer.size = frame.cols * frame.rows * 3;
    img_buffer.virt_addr = frame.data;
    img_buffer.fd = -1;

#if defined(RV1106_1103) 
    // For RV1106/1103, use DMA buffer
    static bool dma_allocated = false;
    static image_buffer_t dma_buffer;
    
    if (!dma_allocated) {
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, img_buffer.size, 
                           &rknn_app_ctx->img_dma_buf.dma_buf_fd, 
                           (void**) &(rknn_app_ctx->img_dma_buf.dma_buf_virt_addr));
        if (ret != 0) {
            printf("Failed to allocate DMA buffer\n");
            return -1;
        }
        
        dma_allocated = true;
        rknn_app_ctx->img_dma_buf.size = img_buffer.size;
        
        dma_buffer.width = img_buffer.width;
        dma_buffer.height = img_buffer.height;
        dma_buffer.format = img_buffer.format;
        dma_buffer.size = img_buffer.size;
        dma_buffer.virt_addr = (unsigned char *)rknn_app_ctx->img_dma_buf.dma_buf_virt_addr;
        dma_buffer.fd = rknn_app_ctx->img_dma_buf.dma_buf_fd;
    }
    
    // Copy image data to DMA buffer
    memcpy(dma_buffer.virt_addr, img_buffer.virt_addr, img_buffer.size);
    dma_sync_cpu_to_device(dma_buffer.fd);
    
    // Run inference with DMA buffer
    ret = inference_yolo11_model(rknn_app_ctx, &dma_buffer, &od_results);
#else
    // Run inference directly with OpenCV buffer
    ret = inference_yolo11_model(rknn_app_ctx, &img_buffer, &od_results);
#endif

    if (ret != 0) {
        printf("inference_yolo11_model fail! ret=%d\n", ret);
        return -1;
    }

    // Draw detection results on the frame
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        float conf = det_result->prop;
        
        // Print detection result
        printf("%s @ (%d %d %d %d) %.3f\n", 
               coco_cls_to_name(det_result->cls_id),
               x1, y1, x2, y2, conf);
        
        // Draw rectangle directly on OpenCV Mat
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), 
                     cv::Scalar(255, 0, 0), 2);
                     
        // Add label with confidence
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), conf * 100);
        cv::putText(frame, text, cv::Point(x1, y1 - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    return 0;
}

// Function to initialize webcam
bool init_webcam(cv::VideoCapture &cap, int webcam_id) {
    printf("Opening webcam %d\n", webcam_id);
    
    // Try different capture APIs in order of preference
    // First try V4L2 (most reliable for Linux)
    if (!cap.open(webcam_id, cv::CAP_V4L2)) {
        printf("Failed to open webcam with V4L2, trying default API\n");
        
        // If V4L2 fails, try the default API
        if (!cap.open(webcam_id)) {
            printf("Failed to open webcam %d\n", webcam_id);
            return false;
        }
    }
    
    // Configure webcam properties
    // Try different resolutions in order of preference
    const int resolution_options[][2] = {
        {640, 480},   // VGA
        {1280, 720},  // 720p
        {1920, 1080}, // 1080p
        {320, 240}    // Fallback
    };
    
    bool resolution_set = false;
    for (int i = 0; i < 4 && !resolution_set; i++) {
        int width = resolution_options[i][0];
        int height = resolution_options[i][1];
        
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // Check if settings were applied
        int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        if (actual_width > 0 && actual_height > 0) {
            printf("Successfully set resolution to %dx%d\n", actual_width, actual_height);
            resolution_set = true;
        } else {
            printf("Failed to set resolution to %dx%d, trying next option...\n", width, height);
        }
    }
    
    // Verify the actual settings
    printf("Actual webcam settings:\n");
    printf("  Width: %d\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH));
    printf("  Height: %d\n", (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("  FPS: %.1f\n", cap.get(cv::CAP_PROP_FPS));
    
    // Test read a frame to verify the webcam works
    cv::Mat test_frame;
    if (!cap.read(test_frame) || test_frame.empty()) {
        printf("Failed to read initial test frame from webcam %d\n", webcam_id);
        return false;
    }
    
    printf("Successfully read test frame of size %dx%d from webcam %d\n", 
           test_frame.cols, test_frame.rows, webcam_id);
    
    return true;
}

// Function to open video file
bool open_video_file(cv::VideoCapture &cap, const char *file_path) {
    printf("Opening video file: %s\n", file_path);
    if (!cap.open(file_path)) {
        printf("Failed to open video file: %s\n", file_path);
        return false;
    }
    
    if (!cap.isOpened()) {
        printf("Failed to open video source\n");
        return false;
    }
    
    return true;
}

// Main processing loop
bool process_video_stream(rknn_app_context_t *rknn_app_ctx, cv::VideoCapture &cap, bool is_webcam, int webcam_id, const char *input_source) {
    // Get video properties
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080)
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    printf("Input %s: %dx%d, %.2f fps\n", 
           is_webcam ? "webcam" : "video", 
           frame_width, frame_height, fps);
    
    // Create window for display
    std::string window_name = is_webcam ? 
        "Webcam - YOLO Detection" : "Video - YOLO Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, frame_width, frame_height);
    
    // Timing variables for FPS calculation
    struct timeval time;
    gettimeofday(&time, nullptr);
    long startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    float currentFps = 0;
    long beforeTime = startTime;
    
    // Main processing loop
    while (keep_running) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            if (is_webcam) {
                printf("Failed to read from webcam. Retrying...\n");
                usleep(100000); // 100ms
                continue;
            } else {
                printf("End of video reached\n");
                break;
            }
        }
        
        if (frame.empty()) {
            printf("Empty frame received\n");
            continue;
        }
        
        // Convert BGR to RGB for YOLO model
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        
        // Process the frame
        process_frame(rknn_app_ctx, frame);
        
        // Convert back to BGR for display
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        
        // Calculate and display FPS
        frames++;
        if (frames % 30 == 0) {
            gettimeofday(&time, nullptr);
            long currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            float interval = float(currentTime - beforeTime) / 1000.0;
            
            if (interval > 0) {
                currentFps = 30.0 / interval;
                beforeTime = currentTime;
            }
        }
        
        // Draw FPS on frame
        char fps_text[32];
        sprintf(fps_text, "FPS: %.1f", currentFps);
        cv::putText(frame, fps_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Add webcam ID or video source information
        char source_text[64];
        if (is_webcam) {
            sprintf(source_text, "Webcam: %d", webcam_id);
        } else {
            sprintf(source_text, "Video: %s", input_source ? input_source : "");
        }
        cv::putText(frame, source_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Display the frame
        cv::imshow(window_name, frame);
        
        // Check for exit key
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) // 'q' or ESC
            break;
        
        // Save frame periodically (every 30 frames)
        if (frames % 30 == 0) {
            char filename[32];
            sprintf(filename, "frame_%04d.jpg", frames / 30);
            cv::imwrite(filename, frame);
        }
    }
    
    // Calculate overall statistics
    gettimeofday(&time, nullptr);
    long endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    float totalTime = float(endTime - startTime) / 1000.0;
    float avgFps = totalTime > 0 ? float(frames) / totalTime : 0;
    
    printf("Summary:\n");
    printf("  Total frames processed: %d\n", frames);
    printf("  Total time: %.2f seconds\n", totalTime);
    printf("  Average FPS: %.2f\n", avgFps);
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return true;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *input_source = NULL;
    bool is_webcam = false;
    int webcam_id = 0;
    int ret = 0;
    
    // Parse command line arguments
    if (argc < 2 || argc > 3) {
        printf("Usage:\n");
        printf("  %s <model_path>                    - Process webcam 0\n", argv[0]);
        printf("  %s <model_path> <webcam_id>        - Process specific webcam\n", argv[0]);
        printf("  %s <model_path> <video_path>       - Process video file\n", argv[0]);
        return -1;
    }
    
    model_path = argv[1];
    
    // Default to webcam 0 if no second argument
    if (argc == 2) {
        is_webcam = true;
        webcam_id = 0;
        printf("No webcam specified, using default webcam 0\n");
    } else {
        // Check if second argument is a number (webcam id)
        input_source = argv[2];
        
        // Try to parse as integer for webcam ID
        char* endptr;
        long val = strtol(input_source, &endptr, 10);
        
        if (*endptr == '\0') {
            // Input is a valid number, treat as webcam ID
            is_webcam = true;
            webcam_id = (int)val;
            printf("Detected numeric argument, using as webcam ID: %d\n", webcam_id);
        } else {
            // Input is not a number, check if it's a device path
            if (strncmp(input_source, "/dev/video", 10) == 0) {
                is_webcam = true;
                // Extract number after "/dev/video"
                webcam_id = atoi(input_source + 10);
                printf("Detected device path, using as webcam ID: %d\n", webcam_id);
            } else {
                // Treat as video file
                is_webcam = false;
                printf("Treating argument as video file path\n");
            }
        }
    }
    
    // Print available webcams on the system
    if (is_webcam) {
        printf("Available video devices on the system:\n");
        int sys_result = system("v4l2-ctl --list-devices 2>/dev/null || echo 'v4l2-ctl command not found'");
        if (sys_result != 0) {
            printf("Warning: Command to list video devices returned non-zero: %d\n", sys_result);
        }
        printf("\n");
    }
    
    // Register signal handler for clean exit
    signal(SIGINT, signal_handler);
    
    // Initialize YOLO model
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    init_post_process();
    
    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        ret = -1;
    } else {
        // Open video source and process
        cv::VideoCapture cap;
        bool success = false;
        
        if (is_webcam) {
            success = init_webcam(cap, webcam_id);
        } else {
            success = open_video_file(cap, input_source);
        }
        
        if (success) {
            process_video_stream(&rknn_app_ctx, cap, is_webcam, webcam_id, input_source);
        } else {
            printf("Failed to open video source\n");
            ret = -1;
        }
    }
    
    // Clean up resources
    deinit_post_process();
    
    int release_ret = release_yolo11_model(&rknn_app_ctx);
    if (release_ret != 0) {
        printf("release_yolo11_model fail! ret=%d\n", release_ret);
    }
    
#if defined(RV1106_1103) 
    if (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr != NULL) {
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                    rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
    }
#endif
    
    return ret;
}
