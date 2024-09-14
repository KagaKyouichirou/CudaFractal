#pragma once

#include <cuda_runtime.h>

#include <QImage>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

class OpenGLWidget: public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit OpenGLWidget(QWidget* parent = nullptr);

protected:
    void initializeGL() override final;
    void resizeGL(int w, int h) override final;
    void paintGL() override final;

private:
    GLuint texture;

    GLuint pbo;
    cudaGraphicsResource* cudaPboResource;

    QOpenGLShaderProgram* shaderProgram;
    QOpenGLBuffer vbo;
    int attrVertex;
    int unifProjMatrix;
    int unifPoints;

    QMatrix4x4 projMatrix;
};
