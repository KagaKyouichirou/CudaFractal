#pragma once

#include <vector_types.h>

extern "C" {
    void launchMandelbrotKernel(
        dim3 dGrid,
        dim3 dBlock,
        cudaSurfaceObject_t surf,
        double oX,
        double oY,
        double step,
        uint16_t limit
    );
}