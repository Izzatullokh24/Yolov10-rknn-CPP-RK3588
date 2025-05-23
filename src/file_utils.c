#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "file_utils.h"

// Define a debug flag (set to 0 to disable debug prints)
#define DEBUG_PRINT 0

// Debug print function that only prints if DEBUG_PRINT is enabled
#define debug_printf(fmt, ...) \
    do { if (DEBUG_PRINT) printf(fmt, ##__VA_ARGS__); } while (0)
  

int read_data_from_file(const char *filename, char **data)
{
    FILE *fp;
    int size;
    struct stat st;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Open file %s failed.\n", filename);
        return -1;
    }

    stat(filename, &st);
    size = st.st_size;

    *data = (char *)malloc(size);
    if (*data == NULL) {
        printf("Malloc buffer failed!\n");
        fclose(fp);
        return -1;
    }

    size_t read_size = fread(*data, 1, size, fp);
    if (read_size != size) {
        printf("Read file %s failed, expected %d bytes but got %ld bytes.\n", 
               filename, size, read_size);
        free(*data);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return size;
}

// Change the signature to match the header
int write_data_to_file(const char *path, const char *data, unsigned int size)
{
    FILE *fp;
    fp = fopen(path, "wb");
    if (fp == NULL) {
        printf("Open file %s failed.\n", path);
        return -1;
    }

    fwrite(data, 1, size, fp);
    fclose(fp);
    return 0;
}
