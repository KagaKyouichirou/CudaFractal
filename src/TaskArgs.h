#pragma once

#include "cuda/FixedPoint8U30Naive.h"

#include <vector_types.h>
#include <cstdint>

struct TaskArgs
{
    dim3 dGrid;
    dim3 dBlock;
    FixedPoint8U30Naive x;
    FixedPoint8U30Naive y;
    FixedPoint8U30Naive h;
    int limit;
};