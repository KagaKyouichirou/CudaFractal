#include "ExportingManagement.h"

#include <QImage>
#include <QOpenGLFramebufferObject>

namespace AppConf
{
extern char const* SHADER_TEXTURE_EXPORT_VERTEX;
extern char const* SHADER_TEXTURE_EXPORT_FRAGMENT;
}  // namespace AppConf

ExportingManager::ExportingManager():
    QObject(nullptr),
    uThread(std::make_unique<QThread>()),
    pExporter(new ImageExporter()),
    uSurface(std::make_unique<QOffscreenSurface>()),
    flagBusy(true)
{
    pExporter->moveToThread(uThread.get());
    // clang-format off
    connect(
        uThread.get(), &QThread::finished,
        pExporter, &ImageExporter::deleteLater,
        Qt::DirectConnection
    );
    connect(
        this, &ExportingManager::signalContextInit,
        pExporter, &ImageExporter::slotContextInit,
        Qt::QueuedConnection
    );
    // clang-format on
    uSurface->create();

    uThread->start();
}

ExportingManager::~ExportingManager()
{
    uThread->quit();
    uThread->wait();

    uSurface->destroy();
}

void ExportingManager::initialize(QOpenGLContext* context)
{
    auto ctx = new QOpenGLContext(nullptr);
    ctx->setShareContext(context);
    ctx->create();
    ctx->moveToThread(uThread.get());
    emit signalContextInit(ctx, uSurface.get());
    // clang-format off
    connect(
        this, &ExportingManager::signalStartExporting,
        pExporter, &ImageExporter::slotStartExporting,
        Qt::QueuedConnection
    );
    connect(
        pExporter, &ImageExporter::signalDoneExporting,
        this, &ExportingManager::slotDoneExporting,
        Qt::QueuedConnection
    );
    // clang-format on
    flagBusy = false;
}

void ExportingManager::requestExporting(std::shared_ptr<TextureScene> scene, ChannelArgs* args, QIODevice* output)
{
    if (flagBusy) {
        return;
    }
    emit signalStartExporting(uSurface.get(), scene, args, output);
}

void ExportingManager::slotDoneExporting()
{
    qDebug() << "Done Exporting";
    flagBusy = false;
}

ImageExporter::ImageExporter(): QObject(nullptr), uContext(), uShader() {}

void ImageExporter::slotContextInit(QOpenGLContext* ctx, QOffscreenSurface* surface)
{
    uContext.reset(ctx);
    uContext->makeCurrent(surface);
    initializeOpenGLFunctions();

    QList<GLfloat> vertices{-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    vertexBuffer.create();
    vertexBuffer.bind();
    vertexBuffer.allocate(vertices.constData(), vertices.count() * sizeof(GLfloat));
    vertexBuffer.release();

    uShader = std::make_unique<QOpenGLShaderProgram>(nullptr);
    if (!uShader->addShaderFromSourceCode(QOpenGLShader::Vertex, AppConf::SHADER_TEXTURE_EXPORT_VERTEX)) {
        qDebug() << "Vertex Shader Failed";
        return;
    }
    if (!uShader->addShaderFromSourceCode(QOpenGLShader::Fragment, AppConf::SHADER_TEXTURE_EXPORT_FRAGMENT)) {
        qDebug() << "Fragment Shader Failed";
        return;
    }
    if (!uShader->link()) {
        qDebug() << "Shader Program Failed To Link";
        return;
    }

    attrVertexCoord = uShader->attributeLocation("vertexCoord");
    unifPoints = uShader->uniformLocation("points");
    unifLogF = uShader->uniformLocation("logFactor");
    unifLogN = uShader->uniformLocation("logNorm");
    unifSpY = uShader->uniformLocation("splineY");
    unifSpK = uShader->uniformLocation("splineK");

    uContext->doneCurrent();
}

void ImageExporter::slotStartExporting(
    QOffscreenSurface* surface,
    std::shared_ptr<TextureScene> scene,
    ChannelArgs* args,
    QIODevice* output
)
{
    auto w = scene->width();
    auto h = scene->height();
    uContext->makeCurrent(surface);
    QOpenGLFramebufferObject fbo(w, h, QOpenGLFramebufferObject::Attachment::NoAttachment, GL_TEXTURE_2D, GL_RGBA8);
    fbo.bind();

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, w, h);

    scene->bind(0u);
    uShader->bind();
    uShader->enableAttributeArray(attrVertexCoord);

    vertexBuffer.bind();
    uShader->setAttributeBuffer(attrVertexCoord, GL_FLOAT, 0, 2, 0);
    vertexBuffer.release();

    uShader->setUniformValue(unifPoints, 0);
    uShader->setUniformValue(unifLogF, static_cast<float>(args->logFactor));
    uShader->setUniformValue(unifLogN, static_cast<float>(args->logNorm));
    uShader->setUniformValueArray(unifSpY, args->splineY.data(), 7);
    uShader->setUniformValueArray(unifSpK, args->splineK.data(), 7);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    uShader->disableAttributeArray(attrVertexCoord);
    uShader->release();
    scene->release();

    auto image = fbo.toImage(true, 0);
    fbo.release();

    uContext->doneCurrent();

    image.save(output, "PNG", 100);
    output->close();

    emit signalDoneExporting();
}
