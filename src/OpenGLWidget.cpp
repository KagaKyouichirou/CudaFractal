#include "OpenGLWidget.h"

#include <windows.h>
#include <cstdio>

#include <cuda_gl_interop.h>

#include "cuda_kernels.h"
#include "shaderSrc.h"

OpenGLWidget::OpenGLWidget(QWidget* parent):
    QOpenGLWidget(parent),
    translateX(0.0),
    translateY(0.0),
    scale(1.0f),
    texture(std::make_unique<QOpenGLTexture, QOpenGLTexture::Target>(QOpenGLTexture::Target2D)),
    cudaSurfaceResource(nullptr),
    shaderProgram(nullptr)
{}

void OpenGLWidget::mousePressEvent(QMouseEvent* event)
{
    flagDragging = true;
    lastMousePos = devicePixelRatio() * event->position();
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent* event) {
    auto r = devicePixelRatio();
    translateX += r * event->x() - lastMousePos.x();
    translateY -= r * event->y() - lastMousePos.y();
    lastMousePos = r * event->position();
    update();
}

void OpenGLWidget::mouseReleaseEvent(QMouseEvent* event) {
    flagDragging = false;
}

void OpenGLWidget::wheelEvent(QWheelEvent* event) {
    constexpr double factorIn = 1.125;
    constexpr double factorOut = 1.0 / factorIn;
    auto const& p = event->position();
    QPointF coord = QPointF(p.x(), height() - p.y()) * devicePixelRatio();
    coord.rx() -= translateX;
    coord.ry() -= translateY;
    auto scaleNew = scale * (event->angleDelta().y() > 0 ? factorIn : factorOut);
    setScale(scaleNew, coord);
    update();
}

void OpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_TEXTURE_2D);

    dim3 sizeGrid(100, 100);
    dim3 sizeBlock(16, 16);
    int width = sizeGrid.x * sizeBlock.x;
    int height = sizeGrid.y * sizeBlock.y;

    texture->create();
    texture->setFormat(QOpenGLTexture::R32F);
    texture->setSize(width, height);
    texture->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::Float32);
    texture->setMinificationFilter(QOpenGLTexture::Linear);
    texture->setMagnificationFilter(QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);

    cudaGraphicsGLRegisterImage(
        &cudaSurfaceResource, texture->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore
    );

    cudaGraphicsMapResources(1, &cudaSurfaceResource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, cudaSurfaceResource, 0, 0);
    {
        auto status = cudaGetLastError();
        if (cudaSuccess != status) {
            printf("cudaGraphicsSubResourceGetMappedArray Failed: %s\n", cudaGetErrorName(status));
        }
    }

    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = array;
    cudaSurfaceObject_t surf;
    cudaCreateSurfaceObject(&surf, &surfRes);
    if (cudaSuccess != cudaGetLastError()) {
        printf("!#\n");
    }

    double half_step = 3.0 / (sizeGrid.x * sizeBlock.x * 2);
    launchMandelbrotKernel(sizeGrid, sizeBlock, surf, -2.0 + half_step, -1.5 + half_step, half_step * 2, 400);

    cudaDeviceSynchronize();
    printf("!!\n");
    if (cudaSuccess != cudaGetLastError()) {
        printf("!\n");
    }

    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &cudaSurfaceResource);

    shaderProgram = new QOpenGLShaderProgram(this);
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, shaderSrcVertex)) {
        printf("Vertex Shader Failed\n");
        return;
    }
    qDebug() << shaderProgram->log();
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, shaderSrcFragment)) {
        printf("Fragment Shader Failed\n");
        return;
    }
    qDebug() << shaderProgram->log();
    if (!shaderProgram->link()) {
        printf("Shader Program Failed To Link\n");
        return;
    }
    qDebug() << shaderProgram->log();
    attrVertexCoord = shaderProgram->attributeLocation("vertexCoord");
    attrTextureCoord = shaderProgram->attributeLocation("textureCoord");
    unifProjMatrix = shaderProgram->uniformLocation("projMatrix");
    unifPoints = shaderProgram->uniformLocation("points");

    GLfloat w = static_cast<GLfloat>(width);
    GLfloat h = static_cast<GLfloat>(height);
    // clang-format off
    // vertex coord + texture coord
    QList<GLfloat> vertices = {
        0.0f,   0.0f,   0.0f,   0.0f,
        w,      0.0f,   1.0f,   0.0f,
        0.0f,   h,      0.0f,   1.0f,
        w,      h,      1.0f,   1.0f
    };
    // clang-format on
    vbo.create();
    vbo.bind();
    vbo.allocate(vertices.constData(), vertices.count() * sizeof(GLfloat));
    vbo.release();
}

void OpenGLWidget::paintGL()
{
    printf("paintGL\n");
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, viewportW, viewportH);    

    // clamp translate args
    // X
    double scaledW = scale * texture->width();    
    if (scaledW > viewportW) {
        translateX = std::clamp(translateX, viewportW - scaledW, 0.0);
    } else {
        translateX = std::clamp(translateX, 0.0, viewportW - scaledW);
    }
    // Y
    double scaledH = scale * texture->height();
    if (scaledH > viewportH) {
        translateY = std::clamp(translateY, viewportH - scaledH, 0.0);
    } else {
        translateY = std::clamp(translateY, 0.0, viewportH - scaledH);
    }

    printf("%f %f", translateX, translateY);

    projMatrix = QMatrix4x4();
    projMatrix.ortho(0.0, viewportW, 0.0, viewportH, -1.0, 1.0);
    projMatrix.translate(static_cast<int>(translateX), static_cast<int>(translateY));
    projMatrix.scale(scale);
    texture->bind();
    printf("%d\n", glGetError());

    if (!shaderProgram->bind()) {
        printf("Shader Program Failed To Bind\n");
        return;
    }
    shaderProgram->enableAttributeArray(attrVertexCoord);
    shaderProgram->enableAttributeArray(attrTextureCoord);

    vbo.bind();
    shaderProgram->setAttributeBuffer(attrVertexCoord, GL_FLOAT, 0, 2, 4 * sizeof(GLfloat));
    shaderProgram->setAttributeBuffer(attrTextureCoord, GL_FLOAT, 2 * sizeof(GLfloat), 2, 4 * sizeof(GLfloat));
    vbo.release();
    shaderProgram->setUniformValue(unifProjMatrix, projMatrix);

    shaderProgram->setUniformValue(unifPoints, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    shaderProgram->disableAttributeArray(attrVertexCoord);
    shaderProgram->disableAttributeArray(attrTextureCoord);
    shaderProgram->release();
}

void OpenGLWidget::resizeGL(int w, int h)
{
    auto r = devicePixelRatio();
    viewportW = static_cast<int>(w * r);
    viewportH = static_cast<int>(h * r);
}

void OpenGLWidget::setScale(double scaleNew, QPointF coord) {
    if (scaleNew > 4.0) {
        scaleNew = 4.0;
    } else if (scaleNew < 0.25) {
        scaleNew = 0.25;
    }
    coord *= (1.0f - scaleNew / scale);
    scale = scaleNew;
    translateX += coord.x();
    translateY += coord.y();
}
