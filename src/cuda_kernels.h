#pragma once

#include <vector_types.h>
#include <cstdint>

extern "C" {
    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, cudaSurfaceObject_t surf, double corner_x, double corner_y, double step, uint16_t limit
    );

    void launchChessboardKernel(dim3 sizeGrid, dim3 sizeBlock, cudaSurfaceObject_t surf);
}