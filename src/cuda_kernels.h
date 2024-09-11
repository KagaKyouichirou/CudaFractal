#pragma once

#include <vector_types.h>
#include <cstdint>

extern "C" {
    uint16_t* allocPoints(size_t size);
    
    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, uint16_t* points, double corner_x, double corner_y, double step, uint16_t limit
    );

    void launchColoringKernel(int sizeGrid, int sizeBlock, uint16_t* points, uchar4* pixels, uint16_t limit);
}