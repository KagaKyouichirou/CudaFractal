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
    virtual ~RenderingManager() override final;

public:
    void slotAddTask(TaskArgs task);
    void slotTaskFinished();

signals:
    void signalTask(TaskArgs task, cudaGraphicsResource* resource);
    void signalSceneRendered(TextureScene* ts);

private:
    void runTask();

private:
    std::unique_ptr<QThread> uThread;
    std::unique_ptr<Renderer> uRenderer;

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
    virtual ~Renderer() override final = default;

public:
    void slotTask(TaskArgs task, cudaGraphicsResource* resource);

signals:
    void signalTaskFinished();
};