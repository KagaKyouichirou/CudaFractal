#pragma once

#include "TextureScene.h"

#include <QFocusEvent>
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
    ~TextureView() override final = default;

    std::shared_ptr<TextureScene> scene() const;

public:
    void slotSceneRendered(TextureScene* ts);
    void slotZoomToFit();
    void slotZoomToActualSize();

signals:
    void signalGLContextInitialized(QOpenGLContext* ctx);
    void signalUploadUnif(QOpenGLShaderProgram* sh, int unifNormF, int unifNormR, int unifSpY, int unifSpK);

protected:
    void focusOutEvent(QFocusEvent* event) override final;
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
    std::shared_ptr<TextureScene> sScene;
    int viewportW;
    int viewportH;
    bool flagDragging;
    QPointF lastMousePos;

    std::unique_ptr<QOpenGLShaderProgram> uShader;
    QOpenGLBuffer vertexBuffer;
    int attrVertexCoord;
    int attrTextureCoord;
    int unifMatrixProj;
    int unifPoints;
    int unifNormF;
    int unifNormR;
    int unifSpY;
    int unifSpK;
};
