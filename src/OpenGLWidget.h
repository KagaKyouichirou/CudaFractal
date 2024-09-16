#pragma once

#include <cuda_runtime.h>

#include <QImage>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>

class OpenGLWidget: public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit OpenGLWidget(QWidget* parent = nullptr);

protected:
    void mousePressEvent(QMouseEvent* event) override final;
    void mouseMoveEvent(QMouseEvent* event) override final;
    void mouseReleaseEvent(QMouseEvent* event) override final;
    void wheelEvent(QWheelEvent* event) override final;

protected:
    void initializeGL() override final;
    void paintGL() override final;
    void resizeGL(int w, int h) override final;

private:
    void setScale(double scaleNew, QPointF coord);

private:
    int viewportW;
    int viewportH;
    double translateX;
    double translateY;
    double scale;

    bool flagDragging;
    QPointF lastMousePos;

    std::unique_ptr<QOpenGLTexture> texture;

    cudaGraphicsResource_t cudaSurfaceResource;

    QOpenGLShaderProgram* shaderProgram;
    QOpenGLBuffer vbo;
    int attrVertexCoord;
    int attrTextureCoord;
    int unifProjMatrix;
    int unifPoints;

    QMatrix4x4 projMatrix;
};
