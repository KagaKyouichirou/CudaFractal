#pragma once

#include "FixedPoint8U30.h"

#include <vector_types.h>
#include <cstdint>

struct TaskArgs
{
    dim3 dGrid;
    dim3 dBlock;
    /* center.x in signed form */
    FixedPoint8U30 x;
    /* center.y in signed form */
    FixedPoint8U30 y;
    /* half-unit; a positive value */
    FixedPoint8U30 h;
    unsigned int limit;
};