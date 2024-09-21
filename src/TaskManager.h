#pragma once

#include "Renderer.h"
#include "TaskArgs.h"
#include "TextureScene.h"

#include <cuda_runtime.h>

#include <QObject>
#include <QOpenGLTexture>
#include <QThread>

class TaskManager: public QObject
{
    Q_OBJECT

public:
    explicit TaskManager();
    virtual ~TaskManager();

public:
    void slotAddTask(TaskArgs task);
    void slotTaskFinished();

signals:
    void signalRunTask(TaskArgs task, cudaGraphicsResource* resource);
    void signalSceneRendered(TextureScene* ts);

private:
    void runTask();

private:
    std::unique_ptr<QThread> thread;
    std::unique_ptr<Renderer> renderer;

    bool flagBusy;
    std::unique_ptr<QOpenGLTexture> texture;
    cudaGraphicsResource_t surfaceResource;
    TaskArgs queued;
};