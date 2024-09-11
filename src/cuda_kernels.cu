__global__ void mandelbrotKernel(uint16_t* points, double corner_x, double corner_y, double step, uint16_t limit) {
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
    points[j + i * row_size] = k;
}

__global__ void coloringKernel(uint16_t* points, uchar4* pixels, double unit) {
    auto p = blockIdx.x * blockDim.x + threadIdx.x;
    double t = log(static_cast<double>(points[p])) / unit;
    pixels[p].w = 255;
    pixels[p].x = static_cast<unsigned char>(256 * t);
    pixels[p].y = static_cast<unsigned char>(256 * t * t * t);
    pixels[p].z = static_cast<unsigned char>(128 * (1 - t * t * t * t));
}

extern "C" {
    uint16_t* allocPoints(size_t size) {
        uint16_t* ptr = nullptr;
        auto status = cudaMalloc<uint16_t>(&ptr, size);
        return ptr;
    }

    void launchMandelbrotKernel(
        dim3 sizeGrid, dim3 sizeBlock, uint16_t* points, double corner_x, double corner_y, double step, uint16_t limit
    ) {
        mandelbrotKernel<<<sizeGrid, sizeBlock>>>(points, corner_x, corner_y, step, limit);
    }

    void launchColoringKernel(int sizeGrid, int sizeBlock, uint16_t* points, uchar4* pixels, uint16_t limit) {
        double unit = log(static_cast<double>(limit));
        coloringKernel<<<sizeGrid, sizeBlock>>>(points, pixels, unit);
    }
}
