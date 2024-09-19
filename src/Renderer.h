#pragma once

#include "TaskArgs.h"

#include <cuda_runtime.h>

#include <QObject>

class Renderer: public QObject
{
    Q_OBJECT

public:
    explicit Renderer();

public:
    void slotRunTask(TaskArgs task, cudaGraphicsResource* resource);

signals:
    void signalTaskFinished();
};
