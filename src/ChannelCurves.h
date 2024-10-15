#pragma once

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

class ChannelCurves: public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit ChannelCurves();
    ~ChannelCurves() override final = default;

signals:
    void signalUploadUnif(QOpenGLShaderProgram* sh, int unifNormF, int unifNormR, int unifSpY, int unifSpK);

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
    int unifNormF;
    int unifNormR;
    int unifSpY;
    int unifSpK;
    int unifStrokePeak;
    int unifStrokeNorm;
    int unifAspectRatio;
};