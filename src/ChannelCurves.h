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
    void signalUploadSplines(QOpenGLShaderProgram* shader, int unifSplineY, int unifSplineK);

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
    int unifSplineY;
    int unifSplineK;
    int unifStrokePeak;
    int unifStrokeNorm;
};