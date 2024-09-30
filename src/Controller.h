#pragma once

#include "ChannelPane.h"
#include "ExportingManagement.h"
#include "InputPane.h"
#include "RenderingManagement.h"
#include "TextureView.h"

#include <QMainWindow>
#include <QObject>
#include <QOpenGLContext>
#include <QStatusBar>
#include <memory>

class Controller: public QObject
{
    Q_OBJECT

public:
    explicit Controller();
    ~Controller() override final = default;

    void start();

private:
    void slotGLContextInitialized(QOpenGLContext* context);
    void slotExportImage();

private:
    std::unique_ptr<QMainWindow> uMainWindow;
    QStatusBar* pStatusBar;

    InputPane* pInputPane;
    ChannelPane* pChannelPane;
    TextureView* pTextureView;
    std::unique_ptr<RenderingManager> uRenderingManager;
    std::unique_ptr<ExportingManager> uExportingManager;
};