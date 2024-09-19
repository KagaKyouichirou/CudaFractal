#include "Renderer.h"

#include "cuda_kernels.h"

#include <QDebug>
#include <QThread>

Renderer::Renderer(): QObject(nullptr) {}

void Renderer::slotRunTask(TaskArgs task, cudaGraphicsResource* resource)
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
