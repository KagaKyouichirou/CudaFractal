#include "TextureView.h"

#include <QDebug>

namespace AppConf
{
extern QString const TEXTURE_VIEW_STYLE;
extern char const* SHADER_TEXTURE_VIEW_VERTEX;
extern char const* SHADER_TEXTURE_VIEW_FRAGMENT;
}  // namespace AppConf

TextureView::TextureView():
    QOpenGLWidget(nullptr),
    sScene(),
    viewportW(0),
    viewportH(0),
    flagDragging(false),
    lastMousePos(0.0, 0.0),
    uShader(),
    vertexBuffer(QOpenGLBuffer::VertexBuffer)
{
    setStyleSheet(AppConf::TEXTURE_VIEW_STYLE);
    setMouseTracking(true);
    setFocusPolicy(Qt::FocusPolicy::ClickFocus);
}

std::shared_ptr<TextureScene> TextureView::scene() const
{
    return sScene;
}

void TextureView::slotSceneRendered(TextureScene* ts)
{
    sScene.reset(ts);
    auto w = static_cast<GLfloat>(sScene->width());
    auto h = static_cast<GLfloat>(sScene->height());
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
    if (!sScene) {
        return;
    }
    double factorW = static_cast<double>(viewportW) / sScene->width();
    double factorH = static_cast<double>(viewportH) / sScene->height();
    sScene->setScale(std::min(factorW, factorH));
    centerView();
}

void TextureView::slotZoomToActualSize()
{
    if (!sScene) {
        return;
    }
    sScene->setScale(1.0);
    centerView();
}

void TextureView::focusOutEvent(QFocusEvent* event)
{
    flagDragging = false;
    QOpenGLWidget::focusOutEvent(event);
}

void TextureView::mousePressEvent(QMouseEvent* event)
{
    if (!sScene) {
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
    if (!sScene) {
        return;
    }
    if (!flagDragging) {
        return;
    }
    auto r = devicePixelRatio();
    sScene->setTranslateX(sScene->translateX() + r * event->x() - lastMousePos.x());
    sScene->setTranslateY(sScene->translateY() - r * event->y() + lastMousePos.y());
    lastMousePos = r * event->position();
    update();
}

void TextureView::mouseReleaseEvent(QMouseEvent* event)
{
    flagDragging = false;
}

void TextureView::wheelEvent(QWheelEvent* event)
{
    if (!sScene) {
        return;
    }
    constexpr double factorIn = 1.125;
    constexpr double factorOut = 1.0 / factorIn;
    auto const& p = event->position();
    QPointF coord = QPointF(p.x(), height() - p.y()) * devicePixelRatio();
    coord.rx() -= sScene->translateX();
    coord.ry() -= sScene->translateY();
    auto scaleNew = sScene->scale() * (event->angleDelta().y() > 0 ? factorIn : factorOut);
    scaleAt(scaleNew, coord);
    update();
}

void TextureView::initializeGL()
{
    emit signalGLContextInitialized(context());
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

    uShader = std::make_unique<QOpenGLShaderProgram>(nullptr);
    if (!uShader->addShaderFromSourceCode(QOpenGLShader::Vertex, AppConf::SHADER_TEXTURE_VIEW_VERTEX)) {
        qDebug() << "Vertex Shader Failed";
        return;
    }
    if (!uShader->addShaderFromSourceCode(QOpenGLShader::Fragment, AppConf::SHADER_TEXTURE_VIEW_FRAGMENT)) {
        qDebug() << "Fragment Shader Failed";
        return;
    }
    if (!uShader->link()) {
        qDebug() << "Shader Program Failed To Link";
        return;
    }
    attrVertexCoord = uShader->attributeLocation("vertexCoord");
    attrTextureCoord = uShader->attributeLocation("textureCoord");
    unifMatrixProj = uShader->uniformLocation("matrixProj");
    unifPoints = uShader->uniformLocation("points");
    unifLogF = uShader->uniformLocation("logFactor");
    unifLogN = uShader->uniformLocation("logNorm");
    unifSpY = uShader->uniformLocation("splineY");
    unifSpK = uShader->uniformLocation("splineK");
}

void TextureView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, viewportW, viewportH);
    if (!sScene) {
        return;
    }

    // clamp translate args
    // X
    auto tX = sScene->translateX();
    double scaledW = sScene->scale() * sScene->width();
    if (scaledW > viewportW) {
        tX = std::clamp(tX, viewportW - scaledW, 0.0);
    } else {
        tX = std::clamp(tX, 0.0, viewportW - scaledW);
    }
    sScene->setTranslateX(tX);
    // Y
    auto tY = sScene->translateY();
    double scaledH = sScene->scale() * sScene->height();
    if (scaledH > viewportH) {
        tY = std::clamp(tY, viewportH - scaledH, 0.0);
    } else {
        tY = std::clamp(tY, 0.0, viewportH - scaledH);
    }
    sScene->setTranslateY(tY);

    QMatrix4x4 matrixProj;
    matrixProj.ortho(0.0, viewportW, 0.0, viewportH, -1.0, 1.0);
    matrixProj.translate(static_cast<int>(tX), static_cast<int>(tY));
    matrixProj.scale(sScene->scale());

    sScene->bind(0u);

    uShader->bind();
    uShader->enableAttributeArray(attrVertexCoord);
    uShader->enableAttributeArray(attrTextureCoord);

    vertexBuffer.bind();
    uShader->setAttributeBuffer(attrVertexCoord, GL_FLOAT, 0, 2, 4 * sizeof(GLfloat));
    uShader->setAttributeBuffer(attrTextureCoord, GL_FLOAT, 2 * sizeof(GLfloat), 2, 4 * sizeof(GLfloat));
    vertexBuffer.release();
    uShader->setUniformValue(unifMatrixProj, matrixProj);

    uShader->setUniformValue(unifPoints, 0);
    emit signalUploadUnif(uShader.get(), unifLogF, unifLogN, unifSpY, unifSpK);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    uShader->disableAttributeArray(attrVertexCoord);
    uShader->disableAttributeArray(attrTextureCoord);
    uShader->release();

    sScene->release();
}

void TextureView::resizeGL(int w, int h)
{
    auto r = devicePixelRatio();
    w = static_cast<int>(w * r);
    h = static_cast<int>(h * r);
    if (sScene) {
        // pin the center
        sScene->setTranslateX(sScene->translateX() + (w - viewportW) / 2.0);
        sScene->setTranslateY(sScene->translateY() + (h - viewportH) / 2.0);
    }
    viewportW = w;
    viewportH = h;
}

void TextureView::scaleAt(double scaleNew, QPointF coord)
{
    if (scaleNew > 4.0) {
        scaleNew = 4.0;
    } else if (scaleNew < 0.25) {
        scaleNew = 0.25;
    }
    coord *= (1.0f - scaleNew / sScene->scale());
    sScene->setScale(scaleNew);
    sScene->setTranslateX(sScene->translateX() + coord.x());
    sScene->setTranslateY(sScene->translateY() + coord.y());
}

void TextureView::centerView()
{
    sScene->setTranslateX((viewportW - sScene->scale() * sScene->width()) / 2);
    sScene->setTranslateY((viewportH - sScene->scale() * sScene->height()) / 2);
    update();
}
