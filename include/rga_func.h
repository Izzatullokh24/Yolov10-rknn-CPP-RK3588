#ifndef RGA_FUNC_H
#define RGA_FUNC_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// RGA resize function
int rga_resize(image_buffer_t* src_image, image_buffer_t* dst_image, 
               image_rect_t* src_rect, image_rect_t* dst_rect);

#ifdef __cplusplus
}
#endif

#endif /* RGA_FUNC_H */
