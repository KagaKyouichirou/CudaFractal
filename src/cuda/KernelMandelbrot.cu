#include "FixedPoint8U30.h"
#include "TaskArgs.h"

struct ArgData
{
    /* signed */
    FixedPoint8U30 oX;
    /* signed */
    FixedPoint8U30 oY;
    /* positive */
    FixedPoint8U30 step;
};

__global__ void kernelMandelbrot(cudaSurfaceObject_t surf, ArgData const* data, unsigned int limit)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    auto eX = data->step;
    eX.mul(x);
    eX.add(data->oX);
    auto eY = data->step;
    eY.mul(y);
    eY.add(data->oY);
    auto real = FixedPoint8U30::zero();
    auto imgn = FixedPoint8U30::zero();
    int k = 0;
    while (k < limit) {
        real.add(eX);
        real.carry();
        bool sign_real = real.abs();
        imgn.add(eY);
        imgn.carry();
        bool sign_imgn = imgn.abs();
        FixedPoint8U30 real2;
        FixedPoint8U30 imgn2;
        real2.sqr(real);
        imgn2.sqr(imgn);
        FixedPoint8U30 norm2 = real2;
        norm2.add(imgn2);
        norm2.carry();
        if (norm2.escaping()) {
            break;
        }
        // imgn = 2 * imgn * real; turn result into signed form
        imgn.dmul(imgn, real);
        if (sign_real ^ sign_imgn) {
            imgn.flip();
        }
        // real = real2 - imgn2; keep result in signed form
        real.flip(imgn2);
        real.add(real2);
        
        k++;
    }
    surf2Dwrite(static_cast<float>(k) / limit, surf, x * sizeof(float), y, cudaBoundaryModeTrap);
}

extern "C" {
    void launchKernelMandelbrot(cudaSurfaceObject_t surf, void const* pTaskArgs)
    {
        auto task = static_cast<TaskArgs const*>(pTaskArgs);
        ArgData args;
        args.oX = task->h;
        args.oX.mul(task->dGrid.x * task->dBlock.x - 1);
        args.oX.flip();
        args.oX.add(task->x);
        args.oY = task->h;
        args.oY.mul(task->dGrid.y * task->dBlock.y - 1);
        args.oY.flip();
        args.oY.add(task->y);
        args.step = task->h;
        args.step.mul(2);

        ArgData* data = nullptr;
        cudaMalloc(&data, sizeof(ArgData));
        cudaMemcpy(data, &args, sizeof(ArgData), cudaMemcpyHostToDevice);

        kernelMandelbrot<<<task->dGrid, task->dBlock>>>(surf, data, task->limit);

        cudaFree(data);
        data = nullptr;
    }
}
