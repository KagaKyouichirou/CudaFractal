#include "OpenGLWidget.h"

#include <windows.h>
#include <cstdio>

#include <cuda_gl_interop.h>

#include "cuda_kernels.h"

OpenGLWidget::OpenGLWidget(QWidget* parent): QOpenGLWidget(parent) {}

void OpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_TEXTURE_2D);

    dim3 sizeGrid(100, 100);
    dim3 sizeBlock(16, 16);
    int width = sizeGrid.x * sizeBlock.x;
    int height = sizeGrid.y * sizeBlock.y;

    // Create a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);

    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // Map PBO and Run CUDA Kernel
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    uchar4* pixels;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pixels), &size, cudaPboResource);

    uint16_t* points = allocPoints(sizeof(uint16_t) * width * height);
    double half_step = 3.0 / (sizeGrid.x * sizeBlock.x * 2);
    launchMandelbrotKernel(sizeGrid, sizeBlock, points, -2.0 + half_step, 1.5 - half_step, half_step * 2, 400);
    if (cudaSuccess != cudaGetLastError()) {
        printf("!\n");
    }
    launchColoringKernel(sizeGrid.x * sizeGrid.y, sizeBlock.x * sizeBlock.y, points, pixels, 400);
    if (cudaSuccess != cudaGetLastError()) {
        printf("#\n");
    }

    cudaDeviceSynchronize();
    auto status = cudaGetLastError();
    if (cudaSuccess != status) {
        printf("%s\n", cudaGetErrorString(status));
    }

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    // Create an OpenGL texture to display the result
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void OpenGLWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void OpenGLWidget::paintGL() {
    printf("paintGL\n");
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
}
