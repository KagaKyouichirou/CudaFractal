#include "TaskManager.h"

#include <windows.h>

#include <cuda_gl_interop.h>

#include <QDebug>

TaskManager::TaskManager():
    QObject(nullptr),
    thread(std::make_unique<QThread>()),
    renderer(std::make_unique<Renderer>()),
    flagBusy(false),
    texture(),
    surfaceResource(nullptr),
    queued()
{
    renderer->moveToThread(thread.get());

    connect(this, &TaskManager::signalRunTask, renderer.get(), &Renderer::slotRunTask, Qt::QueuedConnection);
    connect(renderer.get(), &Renderer::signalTaskFinished, this, &TaskManager::slotTaskFinished, Qt::QueuedConnection);

    thread->start();
}

TaskManager::~TaskManager()
{
    thread->quit();
    thread->wait();
}

void TaskManager::slotAddTask(TaskArgs task)
{
    queued = task;
    qDebug() << "task queued";
    if (!flagBusy) {
        qDebug() << "idle; issue task right now";
        runTask();
    }
}

void TaskManager::slotTaskFinished()
{
    qDebug() << "task finished";
    cudaGraphicsUnmapResources(1, &surfaceResource);
    cudaGraphicsUnregisterResource(surfaceResource);
    surfaceResource = nullptr;

    auto scene = new TextureScene(std::move(texture));

    emit signalSceneRendered(scene);

    if (queued.limit > 0) {
        qDebug() << "queued task found";
        runTask();
    } else {
        qDebug() << "idle";
        flagBusy = false;
    }
}

void TaskManager::runTask()
{
    flagBusy = true;
    int width = queued.dGrid.x * queued.dBlock.x;
    int height = queued.dGrid.y * queued.dBlock.y;

    texture = std::make_unique<QOpenGLTexture, QOpenGLTexture::Target>(QOpenGLTexture::Target2D);
    texture->create();
    texture->setFormat(QOpenGLTexture::R32F);
    texture->setSize(width, height);
    texture->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::Float32);
    texture->setMinificationFilter(QOpenGLTexture::Linear);
    texture->setMagnificationFilter(QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);

    qDebug() << "Texture Allocated";

    cudaGraphicsGLRegisterImage(
        &surfaceResource, texture->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore
    );
    cudaGraphicsMapResources(1, &surfaceResource);

    emit signalRunTask(queued, surfaceResource);

    queued.limit = 0;
}
