#include "RenderingManagement.h"

#include "cuda_kernels.h"

#include <windows.h>

#include <cuda_gl_interop.h>
#include <QDebug>

RenderingManager::RenderingManager():
    QObject(nullptr),
    uThread(std::make_unique<QThread>()),
    uRenderer(std::make_unique<Renderer>()),
    flagBusy(false),
    uScene(),
    surfaceResource(nullptr),
    queued()
{
    uRenderer->moveToThread(uThread.get());
    // clang-format off
    connect(
        this, &RenderingManager::signalTask,
        uRenderer.get(), &Renderer::slotTask,
        Qt::QueuedConnection
    );
    connect(
        uRenderer.get(), &Renderer::signalTaskFinished,
        this, &RenderingManager::slotTaskFinished,
        Qt::QueuedConnection
    );
    // clang-format on
    uThread->start();
}

RenderingManager::~RenderingManager()
{
    uThread->quit();
    uThread->wait();
}

void RenderingManager::slotAddTask(TaskArgs task)
{
    queued = task;
    qDebug() << "task queued";
    if (!flagBusy) {
        qDebug() << "idle; issue task right now";
        runTask();
    }
}

void RenderingManager::slotTaskFinished()
{
    qDebug() << "task finished";
    cudaGraphicsUnmapResources(1, &surfaceResource);
    cudaGraphicsUnregisterResource(surfaceResource);
    surfaceResource = nullptr;

    emit signalSceneRendered(uScene.release());

    if (queued.limit > 0) {
        qDebug() << "queued task found";
        runTask();
    } else {
        qDebug() << "idle";
        flagBusy = false;
    }
}

void RenderingManager::runTask()
{
    flagBusy = true;
    int width = queued.dGrid.x * queued.dBlock.x;
    int height = queued.dGrid.y * queued.dBlock.y;

    uScene = std::make_unique<TextureScene>(QOpenGLTexture::Target2D);
    uScene->create();
    uScene->setFormat(QOpenGLTexture::R32F);
    uScene->setSize(width, height);
    uScene->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::Float32);
    uScene->setMinificationFilter(QOpenGLTexture::Linear);
    uScene->setMagnificationFilter(QOpenGLTexture::Linear);
    uScene->setWrapMode(QOpenGLTexture::WrapMode::ClampToEdge);

    qDebug() << "Texture Allocated";

    cudaGraphicsGLRegisterImage(
        &surfaceResource, uScene->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore
    );
    cudaGraphicsMapResources(1, &surfaceResource);

    emit signalTask(queued, surfaceResource);

    queued.limit = 0;
}

Renderer::Renderer(): QObject(nullptr) {}

void Renderer::slotTask(TaskArgs task, cudaGraphicsResource* resource)
{
    qDebug() << "Rendering...";

    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = array;
    cudaSurfaceObject_t surf;
    cudaCreateSurfaceObject(&surf, &surfRes);

    double oX = task.x - (task.dGrid.x * task.dBlock.x - 1) * task.h;
    double oY = task.y - (task.dGrid.y * task.dBlock.y - 1) * task.h;
    launchMandelbrotKernel(task.dGrid, task.dBlock, surf, oX, oY, task.h * 2, task.limit);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);

    qDebug() << "Rendering Done";

    emit signalTaskFinished();
}