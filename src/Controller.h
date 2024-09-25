#pragma once

#include "ChannelPane.h"
#include "ExportingManagement.h"
#include "InputPane.h"
#include "RenderingManagement.h"
#include "TextureView.h"

#include <QMainWindow>
#include <QObject>
#include <QOpenGLContext>

class Controller: public QObject
{
    Q_OBJECT

public:
    explicit Controller();
    virtual ~Controller() override final = default;

    void start();

private:
    void slotGLContextInitialized(QOpenGLContext* context);
    void slotExportImage();

private:
    std::unique_ptr<QMainWindow> uMainWindow;

    InputPane* pInputPane;
    ChannelPane* pChannelPane;
    TextureView* pTextureView;
    std::unique_ptr<RenderingManager> uRenderingManager;
    std::unique_ptr<ExportingManager> uExportingManager;
};