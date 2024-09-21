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
    virtual ~ChannelCurves() = default;

signals:
    void signalUploadUnif(QOpenGLShaderProgram* shader, int unifLogNormFactor, int unifSplineY, int unifSplineK);

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
    int unifLogNormFactor;
    int unifSplineY;
    int unifSplineK;
    int unifStrokePeak;
    int unifStrokeNorm;
    int unifAspectRatio;
};