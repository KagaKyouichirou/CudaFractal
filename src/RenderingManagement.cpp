#include "RenderingManagement.h"

#include "cuda/Interface.h"

#include <windows.h>

#include <cuda_gl_interop.h>
#include <QElapsedTimer>

#include <QDebug>

RenderingManager::RenderingManager():
    QObject(nullptr),
    uThread(std::make_unique<QThread>()),
    pRenderer(new Renderer()),
    flagBusy(false),
    uScene(),
    surfaceResource(nullptr),
    queued()
{
    pRenderer->moveToThread(uThread.get());
    // clang-format off
    connect(
        uThread.get(), &QThread::finished,
        pRenderer, &Renderer::deleteLater,
        Qt::DirectConnection
    );
    connect(
        this, &RenderingManager::signalTask,
        pRenderer, &Renderer::slotTask,
        Qt::QueuedConnection
    );
    connect(
        pRenderer, &Renderer::signalTaskFinished,
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
    emit signalStatusTemp(QStringLiteral("Rendering Task Added"));
    if (!flagBusy) {
        runTask();
    }
}

void RenderingManager::slotTaskFinished(double seconds)
{
    cudaGraphicsUnmapResources(1, &surfaceResource);
    cudaGraphicsUnregisterResource(surfaceResource);
    surfaceResource = nullptr;

    auto message = QString::asprintf("Rendering Finished: %d × %d %.3fs", uScene->width(), uScene->height(), seconds);
    emit signalStatusTemp(message);
    emit signalSceneRendered(uScene.release());

    if (queued.limit > 0) {
        runTask();
    } else {
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

    cudaGraphicsGLRegisterImage(
        &surfaceResource,
        uScene->textureId(),
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore
    );
    cudaGraphicsMapResources(1, &surfaceResource);

    emit signalTask(queued, surfaceResource);

    queued.limit = 0;
}

Renderer::Renderer(): QObject(nullptr) {}

void Renderer::slotTask(TaskArgs task, cudaGraphicsResource* resource)
{
    QElapsedTimer timer;
    timer.start();

    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = array;
    cudaSurfaceObject_t surf;
    cudaCreateSurfaceObject(&surf, &surfRes);

    launchKernelMandelbrot(surf, &task);

    //launchKernelMandelbrotWarpWise(surf, &task);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    qDebug() << cudaGetErrorName(err);

    cudaDestroySurfaceObject(surf);

    auto seconds = static_cast<double>(timer.elapsed()) / 1000;

    emit signalTaskFinished(seconds);
}