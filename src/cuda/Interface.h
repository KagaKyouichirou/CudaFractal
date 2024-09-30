#pragma once

#include <vector_types.h>

extern "C" {
    void launchMandelbrotKernel(
        dim3 dGrid,
        dim3 dBlock,
        cudaSurfaceObject_t surf,
        void* oX,
        void* oY,
        void* step,
        int limit
    );
}