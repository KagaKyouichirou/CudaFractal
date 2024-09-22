#pragma once

#include "TextureScene.h"

#include <cuda_runtime.h>
#include <QImage>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>

class TextureView: public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit TextureView();

public:
    void slotSceneRendered(TextureScene* ts);
    void slotZoomToFit();
    void slotZoomToActualSize();

signals:
    void signalUploadUnif(QOpenGLShaderProgram* sh, int unifLogF, int unifLogN, int unifSpY, int unifSpK);

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
    void scaleAt(double scaleNew, QPointF coord);
    void centerView();

private:
    std::unique_ptr<TextureScene> scene;
    int viewportW;
    int viewportH;
    bool flagDragging;
    QPointF lastMousePos;
    
    QOpenGLShaderProgram* shaderProgram;
    QOpenGLBuffer vertexBuffer;
    int attrVertexCoord;
    int attrTextureCoord;
    int unifMatrixProj;
    int unifPoints;
    int unifLogF;
    int unifLogN;
    int unifSpY;
    int unifSpK;
};
