#include "FixedPoint8U30Naive.h"

__global__ void mandelbrotKernel(
    cudaSurfaceObject_t surf,
    FixedPoint8U30Naive oX,
    FixedPoint8U30Naive oY,
    FixedPoint8U30Naive step,
    int limit
)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    auto eX = step;
    eX.mul(x);
    eX.add(oX);
    auto eY = step;
    eY.mul(y);
    eY.add(oY);
    auto real = eX;
    auto imgn = eY;
    auto real2 = real;
    real2.sqr();
    auto imgn2 = imgn;
    imgn2.sqr();
    int k = 0;
    while (k < limit && FixedPoint8U30Naive::checkNorm(real2, imgn2)) {
        imgn.mul(real);
        imgn.dou();
        imgn.add(eY);
        real = imgn2;
        real.flip();
        real.add(real2);
        real.add(eX);
        real2 = real;
        real2.sqr();
        imgn2 = imgn;
        imgn2.sqr();
        k++;
    }
    surf2Dwrite(static_cast<float>(k) / limit, surf, x * sizeof(float), y, cudaBoundaryModeTrap);
}

extern "C" {
    void launchMandelbrotKernel(
        dim3 dGrid,
        dim3 dBlock,
        cudaSurfaceObject_t surf,
        void* oX,
        void* oY,
        void* step,
        int limit
    )
    {
        mandelbrotKernel<<<dGrid, dBlock>>>(
            surf,
            *reinterpret_cast<FixedPoint8U30Naive*>(oX),
            *reinterpret_cast<FixedPoint8U30Naive*>(oY),
            *reinterpret_cast<FixedPoint8U30Naive*>(step),
            limit
        );
    }
}
