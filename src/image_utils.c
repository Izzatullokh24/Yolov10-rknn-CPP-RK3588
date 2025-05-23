#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "image_utils.h"
#include "rga_func.h"

// Define a debug flag (set to 0 to disable debug prints)
#define DEBUG_PRINT 0

// Debug print function that only prints if DEBUG_PRINT is enabled
#define debug_printf(fmt, ...) \
    do { if (DEBUG_PRINT) printf(fmt, ##__VA_ARGS__); } while (0)
    

// Calculate image size based on format and dimensions
int get_image_size(image_buffer_t* image)
{
    if (!image) {
        return 0;
    }

    int size = 0;
    switch (image->format) {
        case IMAGE_FORMAT_RGB888:
            size = image->width * image->height * 3;
            break;
        case IMAGE_FORMAT_RGBA8888:
            size = image->width * image->height * 4;
            break;
        case IMAGE_FORMAT_GRAY8:
            size = image->width * image->height;
            break;
        case IMAGE_FORMAT_YUV420SP_NV12:
        case IMAGE_FORMAT_YUV420SP_NV21:
            size = image->width * image->height * 3 / 2;
            break;
        default:
            printf("Unsupported image format: %d\n", image->format);
            break;
    }
    return size;
}

// Implementation of letterbox function
int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letter_box, char color)
{
    if (!src_image || !dst_image || !letter_box) {
        printf("Invalid parameters in convert_image_with_letterbox\n");
        return -1;
    }

    // Calculate scaling factor for letterboxing
    float scale_w = (float)dst_image->width / src_image->width;
    float scale_h = (float)dst_image->height / src_image->height;
    
    // Use the smaller scaling factor to maintain aspect ratio
    float scale = scale_w < scale_h ? scale_w : scale_h;
    letter_box->scale = scale;
    
    // Calculate scaled dimensions
    int scaled_width = (int)(src_image->width * scale);
    int scaled_height = (int)(src_image->height * scale);
    
    // Calculate padding
    letter_box->x_pad = (dst_image->width - scaled_width) / 2;
    letter_box->y_pad = (dst_image->height - scaled_height) / 2;
    
    // Fill destination image with background color
    memset(dst_image->virt_addr, color, dst_image->size);
    
    // Debug print the dimensions
    debug_printf("Letterbox: src=%dx%d, dst=%dx%d, scaled=%dx%d, padding=(%d,%d)\n",
           src_image->width, src_image->height,
           dst_image->width, dst_image->height,
           scaled_width, scaled_height,
           letter_box->x_pad, letter_box->y_pad);
    
    // Define source and destination regions
    image_rect_t src_rect = {0, 0, src_image->width, src_image->height};
    image_rect_t dst_rect = {
        letter_box->x_pad, 
        letter_box->y_pad, 
        letter_box->x_pad + scaled_width, 
        letter_box->y_pad + scaled_height
    };
    
    // Try to use RGA
    int ret = rga_resize(src_image, dst_image, &src_rect, &dst_rect);
    if (ret == 0) {
        debug_printf("RGA resize successful\n");
        return 0;
    }
    
    printf("RGA resize failed, falling back to CPU implementation\n");
    
    // Simple CPU implementation as fallback
    if (src_image->format == IMAGE_FORMAT_RGB888 && dst_image->format == IMAGE_FORMAT_RGB888) {
        // Simple nearest-neighbor scaling
        for (int y = 0; y < scaled_height; y++) {
            for (int x = 0; x < scaled_width; x++) {
                // Source coordinates
                int src_x = (int)(x / scale);
                int src_y = (int)(y / scale);
                
                // Source pixel index
                int src_idx = (src_y * src_image->width + src_x) * 3;
                
                // Destination pixel index (with padding)
                int dst_idx = ((y + letter_box->y_pad) * dst_image->width + (x + letter_box->x_pad)) * 3;
                
                // Copy RGB values
                dst_image->virt_addr[dst_idx]     = src_image->virt_addr[src_idx];
                dst_image->virt_addr[dst_idx + 1] = src_image->virt_addr[src_idx + 1];
                dst_image->virt_addr[dst_idx + 2] = src_image->virt_addr[src_idx + 2];
            }
        }
        debug_printf("CPU resize successful\n");
        return 0;
    }
    
    printf("Unsupported image format for CPU implementation\n");
    return -1;
}

// Stub implementations for other functions
int read_image(const char* path, image_buffer_t* image) { return -1; }
int write_image(const char* path, const image_buffer_t* image) { return -1; }
int convert_image(image_buffer_t* src_image, image_buffer_t* dst_image, 
                  image_rect_t* src_box, image_rect_t* dst_box, char color) { return -1; }
