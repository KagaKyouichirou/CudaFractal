#pragma once

#include "ChannelPane.h"
#include "InputPane.h"
#include "TaskManager.h"
#include "TextureView.h"

#include <QMainWindow>
#include <QObject>
#include <memory>

class Controller: public QObject
{
    Q_OBJECT

public:
    explicit Controller();
    virtual ~Controller() = default;

    void start();

private:
    std::unique_ptr<QMainWindow> pMainWindow;

    InputPane* pInputPane;
    ChannelPane* pChannelPane;
    TextureView* pTextureView;
    std::unique_ptr<TaskManager> uTaskManager;
};