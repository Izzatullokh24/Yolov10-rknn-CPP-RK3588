#include <stdio.h>
#include <string.h>  // Add missing include for memset
#include "common.h"
#include "rga_func.h"

#ifdef RK3588
#include "im2d.h"
#include "rga.h"

// Define a debug flag (set to 0 to disable debug prints)
#define DEBUG_PRINT 0

// Debug print function that only prints if DEBUG_PRINT is enabled
#define debug_printf(fmt, ...) \
    do { if (DEBUG_PRINT) printf(fmt, ##__VA_ARGS__); } while (0)
    
    
// Simple implementation of RGA resize
int rga_resize(image_buffer_t* src_image, image_buffer_t* dst_image, 
               image_rect_t* src_rect, image_rect_t* dst_rect)
{
    if (!src_image || !dst_image) {
        return -1;
    }
    
    // Create RGA buffers
    rga_buffer_t src_buf;
    rga_buffer_t dst_buf;
    memset(&src_buf, 0, sizeof(src_buf));
    memset(&dst_buf, 0, sizeof(dst_buf));
    
    // Convert image format to RGA format
    int src_format = RK_FORMAT_RGB_888;
    int dst_format = RK_FORMAT_RGB_888;
    
    switch (src_image->format) {
        case IMAGE_FORMAT_RGB888:
            src_format = RK_FORMAT_RGB_888;
            break;
        case IMAGE_FORMAT_RGBA8888:
            src_format = RK_FORMAT_RGBA_8888;
            break;
        default:
            printf("Unsupported source image format for RGA: %d\n", src_image->format);
            return -1;
    }
    
    switch (dst_image->format) {
        case IMAGE_FORMAT_RGB888:
            dst_format = RK_FORMAT_RGB_888;
            break;
        case IMAGE_FORMAT_RGBA8888:
            dst_format = RK_FORMAT_RGBA_8888;
            break;
        default:
            printf("Unsupported destination image format for RGA: %d\n", dst_image->format);
            return -1;
    }
    
    // Source buffer - use the whole source image
    src_buf = wrapbuffer_virtualaddr(src_image->virt_addr, 
                                     src_image->width,
                                     src_image->height,
                                     src_format);
    
    // Destination buffer - use the whole destination image
    dst_buf = wrapbuffer_virtualaddr(dst_image->virt_addr,
                                     dst_image->width,
                                     dst_image->height,
                                     dst_format);
    
    // Use the simplest form of imresize
    int ret = imresize(src_buf, dst_buf);
    if (ret != IM_STATUS_SUCCESS) {
        printf("RGA resize error: %d\n", ret);
        return -1;
    }
    
    return 0;
}

#else
// Stub implementation for non-RK3588 platforms
int rga_resize(image_buffer_t* src_image, image_buffer_t* dst_image, 
               image_rect_t* src_rect, image_rect_t* dst_rect)
{
    printf("RGA not available: platform is not RK3588\n");
    return -1;
}
#endif
