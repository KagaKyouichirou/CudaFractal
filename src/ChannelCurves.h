#pragma once

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

class ChannelCurves: public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit ChannelCurves();

signals:
    void signalUploadColorMatrices(QOpenGLShaderProgram* shader, int unifR, int unifG, int unifB);

protected:
    void initializeGL() override final;
    void paintGL() override final;
    void resizeGL(int w, int h) override final;

private:
    int viewportW;
    int viewportH;

    QOpenGLShaderProgram* shaderProgram;
    QOpenGLBuffer vertexBuffer;

    int attrVertexCoord;
    int unifMatrixColorR;
    int unifMatrixColorG;
    int unifMatrixColorB;
    int unifStrokePeak;
    int unifStrokeNorm;
};