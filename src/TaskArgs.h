#pragma once

#include <stdint.h>
#include <vector_types.h>

struct TaskArgs
{
    dim3 dGrid;
    dim3 dBlock;
    double x;
    double y;
    double h;
    uint16_t limit;
};