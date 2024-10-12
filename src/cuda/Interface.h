#pragma once

#include <vector_types.h>

extern "C" {
    void launchKernelMandelbrot(cudaSurfaceObject_t surf, void const* pTaskArgs);

    void launchKernelMandelbrotWarpWise(cudaSurfaceObject_t surf, void const* pTaskArgs);
}