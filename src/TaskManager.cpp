#include "TaskManager.h"

#include <windows.h>

#include <cuda_gl_interop.h>

#include <QDebug>

TaskManager::TaskManager():
    QObject(nullptr),
    thread(std::make_unique<QThread>()),
    renderer(std::make_unique<Renderer>()),
    flagBusy(false),
    scene(),
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

    emit signalSceneRendered(scene.release());

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

    scene = std::make_unique<TextureScene>(QOpenGLTexture::Target2D);
    scene->create();
    scene->setFormat(QOpenGLTexture::R32F);
    scene->setSize(width, height);
    scene->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::Float32);
    scene->setMinificationFilter(QOpenGLTexture::Linear);
    scene->setMagnificationFilter(QOpenGLTexture::Linear);
    scene->setWrapMode(QOpenGLTexture::WrapMode::ClampToEdge);

    qDebug() << "Texture Allocated";

    cudaGraphicsGLRegisterImage(
        &surfaceResource, scene->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore
    );
    cudaGraphicsMapResources(1, &surfaceResource);

    emit signalRunTask(queued, surfaceResource);

    queued.limit = 0;
}
