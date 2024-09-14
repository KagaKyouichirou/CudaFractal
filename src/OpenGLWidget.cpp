#include "OpenGLWidget.h"

#include <windows.h>
#include <cstdio>

#include <cuda_gl_interop.h>

#include "cuda_kernels.h"
#include "shaderSrc.h"

OpenGLWidget::OpenGLWidget(QWidget* parent): QOpenGLWidget(parent), cudaPboResource(nullptr), shaderProgram(nullptr) {}

void OpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_TEXTURE_RECTANGLE);

    dim3 sizeGrid(100, 100);
    dim3 sizeBlock(8, 8);
    int width = sizeGrid.x * sizeBlock.x;
    int height = sizeGrid.y * sizeBlock.y;

    // Create a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float), nullptr, GL_STATIC_DRAW);
    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    // Map PBO and Run CUDA Kernel
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    float* points = nullptr;
    size_t _size = 0;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&points), &_size, cudaPboResource);
    double half_step = 1.5 / (sizeGrid.x * sizeBlock.x * 2);
    launchMandelbrotKernel(sizeGrid, sizeBlock, points, -2.0 + half_step, 1.5 - half_step, half_step * 2, 400);
    if (cudaSuccess != cudaGetLastError()) {
        printf("!\n");
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
    glBindTexture(GL_TEXTURE_RECTANGLE, texture);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    shaderProgram = new QOpenGLShaderProgram(this);
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, shaderSrcVertex)) {
        printf("Vertex Shader Failed\n");
        return;
    }
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, shaderSrcFragment)) {
        printf("Fragment Shader Failed\n");
        return;
    }
    if (!shaderProgram->link()) {
        printf("Shader Program Failed To Link\n");
        return;
    }
    attrVertex = shaderProgram->attributeLocation("vertex");
    unifProjMatrix = shaderProgram->uniformLocation("projMatrix");
    unifPoints = shaderProgram->uniformLocation("points");

    GLfloat vertices[8] = {
        0.0f,
        0.0f,  // Bottom-left
        static_cast<GLfloat>(width),
        0.0f,  // Bottom-right
        0.0f,
        static_cast<GLfloat>(height),  // Top-left
        static_cast<GLfloat>(width),
        static_cast<GLfloat>(height)  // Top-right
    };
    vbo.create();
    vbo.bind();
    vbo.allocate(&vertices, 8 * sizeof(GLfloat));
    vbo.release();
}

void OpenGLWidget::resizeGL(int w, int h) {
    printf("resize w: %d h: %d\n", w, h);
    auto r = devicePixelRatio();
    glViewport(0, 0, w, h);
    projMatrix = QMatrix4x4();
    projMatrix.ortho(0.0, w, 0.0, h, -1.0, 1.0);
}

void OpenGLWidget::paintGL() {
    printf("paintGL\n");
    //glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_RECTANGLE, texture);
    printf("%d\n", glGetError());

    if (!shaderProgram->bind()) {
        printf("Shader Program Failed To Bind\n");
        return;
    }
    shaderProgram->enableAttributeArray(attrVertex);

    vbo.bind();
    shaderProgram->setAttributeBuffer(attrVertex, GL_FLOAT, 0, 2, 0);
    vbo.release();
    shaderProgram->setUniformValue(unifProjMatrix, projMatrix);

    shaderProgram->setUniformValue(unifPoints, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    shaderProgram->disableAttributeArray(attrVertex);
    shaderProgram->release();
}
