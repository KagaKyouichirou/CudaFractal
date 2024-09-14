#pragma once

#include <vector_types.h>
#include <cstdint>

extern "C" {
    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, float* points, double corner_x, double corner_y, double step, uint16_t limit
    );
}