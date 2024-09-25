__global__ void mandelbrotKernel(cudaSurfaceObject_t surf, double oX, double oY, double step, uint16_t limit)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    double eX = oX + x * step;
    double eY = oY + y * step;
    double real = eX;
    double imgn = eY;
    double real2 = real * real;
    double imgn2 = imgn * imgn;
    uint16_t k = 0;
    while (k < limit && real2 + imgn2 <= 4.0) {
        imgn = 2 * real * imgn + eY;
        real = real2 - imgn2 + eX;
        real2 = real * real;
        imgn2 = imgn * imgn;
        k++;
    }
    surf2Dwrite(static_cast<float>(k) / limit, surf, x * sizeof(float), y, cudaBoundaryModeTrap);
}

extern "C" {
    void launchMandelbrotKernel(
        dim3 dGrid,
        dim3 dBlock,
        cudaSurfaceObject_t surf,
        double oX,
        double oY,
        double step,
        uint16_t limit
    )
    {
        mandelbrotKernel<<<dGrid, dBlock>>>(surf, oX, oY, step, limit);
    }
}
