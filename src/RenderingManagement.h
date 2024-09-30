#pragma once

#include "TaskArgs.h"
#include "TextureScene.h"

#include <cuda_runtime.h>

#include <QObject>
#include <QThread>

// forward declaration
class Renderer;

class RenderingManager: public QObject
{
    Q_OBJECT

public:
    explicit RenderingManager();
    ~RenderingManager() override final;

public:
    void slotAddTask(TaskArgs task);
    void slotTaskFinished(double seconds);

signals:
    void signalTask(TaskArgs task, cudaGraphicsResource* resource);
    void signalSceneRendered(TextureScene* ts);
    void signalStatusTemp(QString const& text, int timeout = 0);

private:
    void runTask();

private:
    std::unique_ptr<QThread> uThread;
    Renderer* pRenderer;

    bool flagBusy;
    std::unique_ptr<TextureScene> uScene;
    cudaGraphicsResource_t surfaceResource;
    TaskArgs queued;
};

class Renderer: public QObject
{
    Q_OBJECT

public:
    explicit Renderer();
    ~Renderer() override final = default;

public:
    void slotTask(TaskArgs task, cudaGraphicsResource* resource);

signals:
    void signalTaskFinished(double seconds);
};