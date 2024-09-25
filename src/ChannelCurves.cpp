#include "ChannelCurves.h"

#include <QDebug>
#include <QtMath>

namespace AppConf
{
extern QString const CHANNEL_CURVES_STYLE;
extern float const CHANNEL_CURVES_STROKE_PEAK;
extern float const CHANNEL_CURVES_STROKE_NORM;
extern char const* SHADER_CHANNEL_CURVES_VERTEX;
extern char const* SHADER_CHANNEL_CURVES_FRAGMENT;
}  // namespace AppConf

ChannelCurves::ChannelCurves():
    QOpenGLWidget(nullptr),
    viewportW(0),
    viewportH(0),
    shaderProgram(nullptr),
    vertexBuffer(QOpenGLBuffer::VertexBuffer)
{
    setStyleSheet(AppConf::CHANNEL_CURVES_STYLE);
}

void ChannelCurves::initializeGL()
{
    initializeOpenGLFunctions();
    //glEnable(GL_TEXTURE_2D);
    QList<GLfloat> vertices{-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    vertexBuffer.create();
    vertexBuffer.bind();
    vertexBuffer.allocate(vertices.constData(), vertices.count() * sizeof(GLfloat));
    vertexBuffer.release();
    shaderProgram = new QOpenGLShaderProgram(this);
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, AppConf::SHADER_CHANNEL_CURVES_VERTEX)) {
        qDebug() << "Vertex Shader Failed";
        return;
    }
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, AppConf::SHADER_CHANNEL_CURVES_FRAGMENT)) {
        qDebug() << "Fragment Shader Failed";
        return;
    }
    if (!shaderProgram->link()) {
        qDebug() << "Shader Program Failed To Link";
        return;
    }
    attrVertexCoord = shaderProgram->attributeLocation("vertexCoord");
    unifLogF = shaderProgram->uniformLocation("logFactor");
    unifLogN = shaderProgram->uniformLocation("logNorm");
    unifSpY = shaderProgram->uniformLocation("splineY");
    unifSpK = shaderProgram->uniformLocation("splineK");
    unifStrokePeak = shaderProgram->uniformLocation("strokePeak");
    unifStrokeNorm = shaderProgram->uniformLocation("strokeNorm");
    unifAspectRatio = shaderProgram->uniformLocation("aspectRatio");
}

void ChannelCurves::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, viewportW, viewportH);
    if (!shaderProgram->bind()) {
        qDebug() << "Shader Program Failed To Bind";
        return;
    }
    shaderProgram->enableAttributeArray(attrVertexCoord);
    vertexBuffer.bind();
    shaderProgram->setAttributeBuffer(attrVertexCoord, GL_FLOAT, 0, 2, 0);
    vertexBuffer.release();

    emit signalUploadUnif(shaderProgram, unifLogF, unifLogN, unifSpY, unifSpK);
    shaderProgram->setUniformValue(unifStrokePeak, AppConf::CHANNEL_CURVES_STROKE_PEAK);
    // stroke-width is fixed in pixels, not relative to viewport size
    auto norm = AppConf::CHANNEL_CURVES_STROKE_NORM * viewportW;
    shaderProgram->setUniformValue(unifStrokeNorm, norm);
    auto aspectRatio = static_cast<float>(viewportH) / viewportW;
    shaderProgram->setUniformValue(unifAspectRatio, aspectRatio);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    shaderProgram->disableAttributeArray(attrVertexCoord);
}

void ChannelCurves::resizeGL(int w, int h)
{
    auto r = devicePixelRatio();
    viewportW = static_cast<int>(w * r);
    viewportH = static_cast<int>(h * r);
}
