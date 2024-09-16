__global__ void mandelbrotKernel(cudaSurfaceObject_t surf, double corner_x, double corner_y, double step, uint16_t limit) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    double e_x = corner_x + x * step;
    double e_y = corner_y + y * step;
    double real = e_x;
    double imgn = e_y;
    double real2 = real * real;
    double imgn2 = imgn * imgn;
    uint16_t k = 0;
    while (k < limit && real2 + imgn2 <= 4.0) {
        imgn = 2 * real * imgn + e_y;
        real = real2 - imgn2 + e_x;
        real2 = real * real;
        imgn2 = imgn * imgn;
        k++;
    }
    surf2Dwrite(static_cast<float>(k) / limit, surf, x * sizeof(float), y, cudaBoundaryModeZero);
}

__global__ void chessboardKernel(cudaSurfaceObject_t surf) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    float flag = static_cast<float>(blockIdx.x + blockIdx.y) / 200;
    surf2Dwrite(flag, surf, x * sizeof(float), y, cudaBoundaryModeTrap);
}

extern "C" {
    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, cudaSurfaceObject_t surf, double corner_x, double corner_y, double step, uint16_t limit
    ) {
        mandelbrotKernel<<<sizeGrid, sizeBlock>>>(surf, corner_x, corner_y, step, limit);
    }

    void launchChessboardKernel(dim3 sizeGrid, dim3 sizeBlock, cudaSurfaceObject_t surf) {
        chessboardKernel<<<sizeGrid, sizeBlock>>>(surf);
    }
}
