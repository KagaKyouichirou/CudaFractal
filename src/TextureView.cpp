#include "TextureView.h"

#include "cuda_kernels.h"

#include <windows.h>

#include <cuda_gl_interop.h>
#include <QDebug>
#include <QtConcurrent>

namespace ProjConf
{
extern QString const TEXTURE_VIEW_STYLE;
extern char const* SHADER_TEXTURE_VIEW_VERTEX;
extern char const* SHADER_TEXTURE_VIEW_FRAGMENT;
}  // namespace ProjConf

TextureView::TextureView():
    QOpenGLWidget(nullptr),
    scene(),
    viewportW(0),
    viewportH(0),
    flagDragging(false),
    lastMousePos(0.0, 0.0),
    shaderProgram(nullptr),
    vertexBuffer(QOpenGLBuffer::VertexBuffer)
{
    setStyleSheet(ProjConf::TEXTURE_VIEW_STYLE);
}

void TextureView::slotSceneRendered(TextureScene* ts)
{
    qDebug() << "Scene Rendered" << ts->width() << ts->height();
    scene.reset(ts);
    auto w = static_cast<GLfloat>(scene->width());
    auto h = static_cast<GLfloat>(scene->height());
    constexpr auto size = sizeof(GLfloat);
    // update the model vertices
    vertexBuffer.bind();
    vertexBuffer.write(4 * size, &w, size);
    vertexBuffer.write(9 * size, &h, size);
    vertexBuffer.write(12 * size, &w, size);
    vertexBuffer.write(13 * size, &h, size);
    vertexBuffer.release();
    slotZoomToActualSize();
}

void TextureView::slotZoomToFit()
{
    if (!scene) {
        return;
    }
    double factorW = static_cast<double>(viewportW) / scene->width();
    double factorH = static_cast<double>(viewportH) / scene->height();
    scene->setScale(std::min(factorW, factorH));
    centerView();
}

void TextureView::slotZoomToActualSize()
{
    if (!scene) {
        return;
    }
    scene->setScale(1.0);
    centerView();
}

void TextureView::mousePressEvent(QMouseEvent* event)
{
    if (!scene) {
        return;
    }
    switch (event->button()) {
        case Qt::MouseButton::LeftButton:
            flagDragging = true;
            lastMousePos = devicePixelRatio() * event->position();
            break;
        case Qt::MouseButton::RightButton:
            slotZoomToFit();
            break;
        case Qt::MouseButton::MiddleButton:
            slotZoomToActualSize();
            break;
        default:
            break;
    }
}

void TextureView::mouseMoveEvent(QMouseEvent* event)
{
    if (!scene) {
        return;
    }
    if (!flagDragging) {
        return;
    }
    auto r = devicePixelRatio();
    scene->setTranslateX(scene->translateX() + r * event->x() - lastMousePos.x());
    scene->setTranslateY(scene->translateY() - r * event->y() + lastMousePos.y());
    lastMousePos = r * event->position();
    update();
}

void TextureView::mouseReleaseEvent(QMouseEvent* event)
{
    flagDragging = false;
}

void TextureView::wheelEvent(QWheelEvent* event)
{
    if (!scene) {
        return;
    }
    constexpr double factorIn = 1.125;
    constexpr double factorOut = 1.0 / factorIn;
    auto const& p = event->position();
    QPointF coord = QPointF(p.x(), height() - p.y()) * devicePixelRatio();
    coord.rx() -= scene->translateX();
    coord.ry() -= scene->translateY();
    auto scaleNew = scene->scale() * (event->angleDelta().y() > 0 ? factorIn : factorOut);
    scaleAt(scaleNew, coord);
    update();
}

void TextureView::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_TEXTURE_2D);

    // clang-format off
    // vertex coord + texture coord
    QList<GLfloat> vertices = {
        0.0f,   0.0f,   0.0f,   0.0f,
        1.0f,   0.0f,   1.0f,   0.0f,
        0.0f,   1.0f,   0.0f,   1.0f,
        1.0f,   1.0f,   1.0f,   1.0f
    };
    // clang-format on
    vertexBuffer.create();
    vertexBuffer.bind();
    vertexBuffer.allocate(vertices.constData(), vertices.count() * sizeof(GLfloat));
    vertexBuffer.release();

    shaderProgram = new QOpenGLShaderProgram(this);
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, ProjConf::SHADER_TEXTURE_VIEW_VERTEX)) {
        qDebug() << "Vertex Shader Failed";
        return;
    }
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, ProjConf::SHADER_TEXTURE_VIEW_FRAGMENT)) {
        qDebug() << "Fragment Shader Failed";
        return;
    }
    if (!shaderProgram->link()) {
        qDebug() << "Shader Program Failed To Link";
        return;
    }
    attrVertexCoord = shaderProgram->attributeLocation("vertexCoord");
    attrTextureCoord = shaderProgram->attributeLocation("textureCoord");
    unifMatrixProj = shaderProgram->uniformLocation("matrixProj");
    unifPoints = shaderProgram->uniformLocation("points");
    unifSplineY = shaderProgram->uniformLocation("splineY");
    unifSplineK = shaderProgram->uniformLocation("splineK");
}

void TextureView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, viewportW, viewportH);
    if (!scene) {
        return;
    }

    // clamp translate args
    // X
    auto tX = scene->translateX();
    double scaledW = scene->scale() * scene->width();
    if (scaledW > viewportW) {
        tX = std::clamp(tX, viewportW - scaledW, 0.0);
    } else {
        tX = std::clamp(tX, 0.0, viewportW - scaledW);
    }
    scene->setTranslateX(tX);
    // Y
    auto tY = scene->translateY();
    double scaledH = scene->scale() * scene->height();
    if (scaledH > viewportH) {
        tY = std::clamp(tY, viewportH - scaledH, 0.0);
    } else {
        tY = std::clamp(tY, 0.0, viewportH - scaledH);
    }
    scene->setTranslateY(tY);

    QMatrix4x4 matrixProj;
    matrixProj.ortho(0.0, viewportW, 0.0, viewportH, -1.0, 1.0);
    matrixProj.translate(static_cast<int>(tX), static_cast<int>(tY));
    matrixProj.scale(scene->scale());

    scene->bindTexture();

    if (!shaderProgram->bind()) {
        qDebug() << "Shader Program Failed To Bind";
        return;
    }
    shaderProgram->enableAttributeArray(attrVertexCoord);
    shaderProgram->enableAttributeArray(attrTextureCoord);

    vertexBuffer.bind();
    shaderProgram->setAttributeBuffer(attrVertexCoord, GL_FLOAT, 0, 2, 4 * sizeof(GLfloat));
    shaderProgram->setAttributeBuffer(attrTextureCoord, GL_FLOAT, 2 * sizeof(GLfloat), 2, 4 * sizeof(GLfloat));
    vertexBuffer.release();
    shaderProgram->setUniformValue(unifMatrixProj, matrixProj);

    shaderProgram->setUniformValue(unifPoints, 0);
    emit signalUploadSplines(shaderProgram, unifSplineY, unifSplineK);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    shaderProgram->disableAttributeArray(attrVertexCoord);
    shaderProgram->disableAttributeArray(attrTextureCoord);
    shaderProgram->release();

    scene->releaseTexture();
}

void TextureView::resizeGL(int w, int h)
{
    auto r = devicePixelRatio();
    viewportW = static_cast<int>(w * r);
    viewportH = static_cast<int>(h * r);
}

void TextureView::scaleAt(double scaleNew, QPointF coord)
{
    if (scaleNew > 4.0) {
        scaleNew = 4.0;
    } else if (scaleNew < 0.25) {
        scaleNew = 0.25;
    }
    coord *= (1.0f - scaleNew / scene->scale());
    scene->setScale(scaleNew);
    scene->setTranslateX(scene->translateX() + coord.x());
    scene->setTranslateY(scene->translateY() + coord.y());
}

void TextureView::centerView()
{
    scene->setTranslateX((viewportW - scene->scale() * scene->width()) / 2);
    scene->setTranslateY((viewportH - scene->scale() * scene->height()) / 2);
    update();
}
