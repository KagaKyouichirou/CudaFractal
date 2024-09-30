#pragma once

#include "ChannelArgs.h"
#include "TextureScene.h"

#include <QIODevice>
#include <QObject>
#include <QOffscreenSurface>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QThread>

// forward declaration
class ImageExporter;

class ExportingManager: public QObject
{
    Q_OBJECT

public:
    explicit ExportingManager();
    ~ExportingManager() override final;

    void initialize(QOpenGLContext* context);
    void requestExporting(std::shared_ptr<TextureScene> scene, ChannelArgs* args, QIODevice* output);

public:
    void slotDoneExporting();

signals:
    void signalContextInit(QOpenGLContext* context, QOffscreenSurface* surface);
    void signalStartExporting(
        QOffscreenSurface* surface,
        std::shared_ptr<TextureScene> scene,
        ChannelArgs* args,
        QIODevice* output
    );

private:
    std::unique_ptr<QThread> uThread;
    ImageExporter* pExporter;

    std::unique_ptr<QOffscreenSurface> uSurface;

    bool flagBusy;
};

class ImageExporter: public QObject, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit ImageExporter();
    ~ImageExporter() override final = default;

public:
    void slotContextInit(QOpenGLContext* context, QOffscreenSurface* surface);
    void slotStartExporting(
        QOffscreenSurface* surface,
        std::shared_ptr<TextureScene> scene,
        ChannelArgs* args,
        QIODevice* output
    );

signals:
    void signalDoneExporting();

private:
    std::unique_ptr<QOpenGLContext> uContext;
    std::unique_ptr<QOpenGLShaderProgram> uShader;
    QOpenGLBuffer vertexBuffer;
    int attrVertexCoord;
    int unifPoints;
    int unifLogF;
    int unifLogN;
    int unifSpY;
    int unifSpK;
};