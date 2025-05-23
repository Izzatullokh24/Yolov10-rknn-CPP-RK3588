#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <thread>
#include <cstdlib>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov10.hpp"  // Using our adapter class
#include "rknnPool.hpp"


int calculateOptimalThreads(int total_instances = 1) {
    // Get total CPU cores
    int total_cores = std::thread::hardware_concurrency();
    printf("Detected %d CPU cores\n", total_cores);
    
    // Reserve 25% of cores for system processes (minimum 2)
    int reserved_cores = std::max(2, total_cores / 4);
    int available_cores = total_cores - reserved_cores;
    
    // Calculate threads per instance
    int threads_per_instance = std::max(1, available_cores / total_instances);
    
    // Cap at reasonable maximum for NPU
    int max_threads_per_npu = 6; // Reduced from 8 for 1080p processing
    threads_per_instance = std::min(threads_per_instance, max_threads_per_npu);
    
    printf("Thread calculation: %d total cores, %d reserved, %d available\n", 
           total_cores, reserved_cores, available_cores);
    printf("Recommended threads per instance: %d (for %d instances)\n", 
           threads_per_instance, total_instances);
    
    return threads_per_instance;
}

int main(int argc, char **argv)
{
    char *model_name = NULL;
    char *input_source = NULL;
    bool is_webcam = false;
    int webcam_id = 0;
    int total_instances = 1; // Default to 1 if not specified
    
    if (argc < 3 || argc > 4)
    {
        printf("Usage: %s <rknn model> <input source> [total_instances]\n", argv[0]);
        printf("For webcam: Use 'webcam:X' where X is the camera ID (usually 0 for built-in webcam)\n");
        printf("For video: Use the path to the video file\n");
        printf("total_instances: How many instances of this program will run simultaneously (optional)\n");
        printf("\nExample for 4 webcams:\n");
        printf("  Terminal 1: %s model.rknn webcam:0 4\n", argv[0]);
        printf("  Terminal 2: %s model.rknn webcam:1 4\n", argv[0]);
        printf("  Terminal 3: %s model.rknn webcam:2 4\n", argv[0]);
        printf("  Terminal 4: %s model.rknn webcam:3 4\n", argv[0]);
        return -1;
    }
    
    // Parameter 2: Model path
    model_name = (char *)argv[1];
    
    // Parameter 3: Video source or webcam identifier
    input_source = argv[2];
    
    // Parameter 4: Total instances (optional)
    if (argc == 4) {
        total_instances = std::atoi(argv[3]);
        if (total_instances <= 0) {
            printf("Error: total_instances must be a positive integer\n");
            return -1;
        }
    }
    
    // Check if input source is webcam
    std::string input_str(input_source);
    if (input_str.find("webcam:") == 0) {
        is_webcam = true;
        std::string webcam_id_str = input_str.substr(7); // Extract ID after "webcam:"
        try {
            webcam_id = std::stoi(webcam_id_str);
        } catch (const std::exception& e) {
            printf("Invalid webcam ID format. Use 'webcam:X' where X is an integer.\n");
            return -1;
        }
    }
    
    // Calculate optimal thread number based on total instances
    int threadNum = calculateOptimalThreads(total_instances);
    
    // Allow override via environment variable (useful for fine-tuning)
    const char* thread_override = getenv("RKNN_THREADS");
    if (thread_override) {
        threadNum = std::atoi(thread_override);
        printf("Thread count overridden by RKNN_THREADS environment variable: %d\n", threadNum);
    }
    
    printf("Using %d threads for RKNN thread pool\n", threadNum);
    
    // Initialize rknn thread pool with calculated thread number
    // Using the rkYolov10 adapter class
    rknnPool<rkYolov10, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool initialization failed!\n");
        return -1;
    }
    
    cv::VideoCapture capture;
    
    // Open webcam or video file with additional error checking
    if (is_webcam) {
        printf("Opening webcam with ID: %d\n", webcam_id);
        if (!capture.open(webcam_id, cv::CAP_V4L2)) {
            printf("Failed to open webcam %d with CAP_V4L2\n", webcam_id);
            return -1;
        }
        
        // Set codec to MJPEG for better performance with webcams
        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        
        // Adjust resolution based on number of instances for better performance
        int target_width = 1920;
        int target_height = 1080;
        
        // Scale down resolution if running multiple instances
        if (total_instances > 1) {
            if (total_instances <= 4) {
                target_width = 1280;
                target_height = 720;
                printf("Adjusted resolution to 720p for multiple instances\n");
            } else {
                target_width = 640;
                target_height = 480;
                printf("Adjusted resolution to 480p for many instances\n");
            }
        }
        
        capture.set(cv::CAP_PROP_FRAME_WIDTH, target_width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, target_height);
        capture.set(cv::CAP_PROP_FPS, 30);
        
        // Verify actual settings
        int actual_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int actual_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = capture.get(cv::CAP_PROP_FPS);
        
        printf("Webcam %d settings - Requested: %dx%d@30fps, Actual: %dx%d@%.1ffps\n",
               webcam_id, target_width, target_height, actual_width, actual_height, actual_fps);
        
    } else {
        printf("Opening video file: %s\n", input_source);
        if (!capture.open(input_source)) {
            printf("Failed to open video file: %s\n", input_source);
            return -1;
        }
    }
    
    // Verify that the video was opened properly
    if (!capture.isOpened()) {
        printf("Error: Video source could not be opened properly!\n");
        return -1;
    }
    
    // Get video properties for display with error checking
    int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(cv::CAP_PROP_FPS);
    
    if (frame_width <= 0 || frame_height <= 0) {
        printf("Warning: Could not detect valid frame dimensions. Using defaults.\n");
        frame_width = 640;
        frame_height = 480;
    }
    
    printf("Input %s: %dx%d, %.2f fps\n", 
           is_webcam ? "webcam" : "video", 
           frame_width, frame_height, 
           fps);
    
    // Create display window with unique name for each webcam
    std::string window_name = is_webcam ? 
        ("Webcam " + std::to_string(webcam_id) + " - YOLO v10") : 
        ("Video - YOLO v10");
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    
    // Adjust display window size based on input resolution and instance count
    int display_width = std::min(frame_width, total_instances > 2 ? 960 : 1920);
    int display_height = std::min(frame_height, total_instances > 2 ? 540 : 1080);
    cv::resizeWindow(window_name, display_width, display_height);
    
    // Position windows in a grid pattern for multiple instances
    if (is_webcam && total_instances > 1) {
        int grid_cols = (total_instances <= 4) ? 2 : 3;
        int window_x = (webcam_id % grid_cols) * (display_width + 50);
        int window_y = (webcam_id / grid_cols) * (display_height + 80);
        cv::moveWindow(window_name, window_x, window_y);
        printf("Positioned window at (%d, %d)\n", window_x, window_y);
    }
    
    // Timing variables
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    auto beforeTime = startTime;
    int fps_display_interval = 30;
    
    printf("Press 'q' to exit\n");
    
    // Read first frame to verify the video stream works
    cv::Mat test_frame;
    if (!capture.read(test_frame) || test_frame.empty()) {
        printf("Error: Could not read first frame from video source!\n");
        return -1;
    }
    
    // Check if the dimensions make sense
    printf("First frame dimensions: %dx%d, channels: %d\n", 
           test_frame.cols, test_frame.rows, test_frame.channels());
    
    if (test_frame.channels() != 3) {
        printf("Warning: Expected 3-channel BGR image, got %d channels\n", test_frame.channels());
        // Convert to BGR if needed
        if (test_frame.channels() == 1) {
            cv::cvtColor(test_frame, test_frame, cv::COLOR_GRAY2BGR);
            printf("Converting grayscale to BGR\n");
        } else if (test_frame.channels() == 4) {
            cv::cvtColor(test_frame, test_frame, cv::COLOR_BGRA2BGR);
            printf("Converting BGRA to BGR\n");
        }
    }
    
    // Reset the video back to the beginning for video files
    if (!is_webcam) {
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
    
    // Main processing loop with better error handling
    while (true)
    {
        cv::Mat img;
        if (!capture.read(img))
        {
            if (is_webcam) {
                printf("Failed to read frame from webcam %d. Retrying...\n", webcam_id);
                cv::waitKey(30); // Wait a bit before retrying
                continue;
            } else {
                printf("End of video file reached\n");
                break;
            }
        }
        
        if (img.empty()) {
            printf("Warning: Empty frame received from webcam %d. Skipping.\n", webcam_id);
            continue;
        }
        
        // Verify frame dimensions and format
        if (img.type() != CV_8UC3) {
            printf("Converting image to proper format\n");
            cv::cvtColor(img, img, img.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
        }
        
        // Ensure reasonable dimensions for RKNN processing
        if (img.cols > 1920 || img.rows > 1080) {
            cv::resize(img, img, cv::Size(std::min(img.cols, 1920), std::min(img.rows, 1080)));
        }
        
        // Submit frame to thread pool for processing
        if (testPool.put(img) != 0) {
            printf("Failed to put frame in thread pool for webcam %d. Queue might be full.\n", webcam_id);
            // Instead of breaking, just skip a frame
            cv::waitKey(10);
            continue;
        }
        
        // Get processed frame if we have processed enough frames to have results
        if (frames >= threadNum) {
            cv::Mat result;
            int get_result = testPool.get(result);
            if (get_result != 0) {
                printf("Warning: Failed to get processed frame from thread pool (code: %d)\n", get_result);
                // Don't break, just continue trying
            } else if (!result.empty()) {
                img = result;
            }
        }
        
        // Display the frame
        cv::imshow(window_name, img);
        
        // Check for exit key
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) // 'q' or ESC key
            break;
        
        frames++;
        
        // Calculate and display FPS periodically
        if (frames % fps_display_interval == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            float interval = float(currentTime - beforeTime) / 1000.0;
            
            if (interval > 0) {
                float current_fps = fps_display_interval / interval;
                printf("Webcam %d - Current FPS: %.2f\n", webcam_id, current_fps);
                
                // Draw FPS on frame (if not empty)
                if (!img.empty()) {
                    std::string fps_text = "Cam" + std::to_string(webcam_id) + " FPS: " + std::to_string(int(current_fps));
                    cv::putText(img, fps_text, cv::Point(20, 40), 
                              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    
                    // Also show thread count
                    std::string thread_text = "Threads: " + std::to_string(threadNum);
                    cv::putText(img, thread_text, cv::Point(20, 80), 
                              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                }
            }
            
            beforeTime = currentTime;
        }
    }
    
    // Process remaining frames in thread pool with better error handling
    printf("Processing remaining frames in thread pool...\n");
    int remaining_frames = 0;
    
    while (remaining_frames < threadNum) // Only try for the number of frames we expect
    {
        cv::Mat img;
        int get_status = testPool.get(img);
        
        if (get_status != 0) {
            printf("Warning: Error getting frame from pool during cleanup (code: %d)\n", get_status);
            remaining_frames++; // Count as processed even if error
            continue;
        }
        
        if (!img.empty()) {
            cv::imshow(window_name, img);
            if (cv::waitKey(1) == 'q')
                break;
            frames++;
        }
        
        remaining_frames++;
    }
    
    // Calculate overall statistics
    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    float total_time_seconds = float(endTime - startTime) / 1000.0;
    float avg_fps = total_time_seconds > 0 ? float(frames) / total_time_seconds : 0;
    
    printf("Summary for webcam %d:\n", webcam_id);
    printf("  Total frames processed: %d\n", frames);
    printf("  Total time: %.2f seconds\n", total_time_seconds);
    printf("  Average FPS: %.2f\n", avg_fps);
    printf("  Threads used: %d\n", threadNum);
    printf("  Resolution: %dx%d\n", frame_width, frame_height);
    
    // Release resources
    printf("Releasing resources...\n");
    capture.release();
    cv::destroyAllWindows();
    
    printf("Program completed successfully\n");
    return 0;
}
