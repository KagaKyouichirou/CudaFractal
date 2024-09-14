__global__ void mandelbrotKernel(float* points, double corner_x, double corner_y, double step, uint16_t limit) {
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    double e_x = corner_x + j * step;
    double e_y = corner_y - i * step;
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
    int row_size = gridDim.x * blockDim.x;
    points[j + i * row_size] = static_cast<float>(k) / limit;
}

extern "C" {
    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, float* points, double corner_x, double corner_y, double step, uint16_t limit
    ) {
        mandelbrotKernel<<<sizeGrid, sizeBlock>>>(points, corner_x, corner_y, step, limit);
    }
}
