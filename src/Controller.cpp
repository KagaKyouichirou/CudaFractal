#include "Controller.h"

#include <QFile>
#include <QFileDialog>
#include <QSplitter>
#include <QTabWidget>

namespace AppConf
{
extern QSize const DEFAULT_MAINWINDOW_SIZE;
}  // namespace AppConf

Controller::Controller():
    QObject(nullptr),
    uMainWindow(std::make_unique<QMainWindow>(nullptr, Qt::Window)),
    pStatusBar(new QStatusBar()),
    pInputPane(new InputPane()),
    pChannelPane(new ChannelPane()),
    pTextureView(new TextureView()),
    uRenderingManager(std::make_unique<RenderingManager>()),
    uExportingManager(std::make_unique<ExportingManager>())
{
    auto tabs = new QTabWidget(nullptr);
    tabs->addTab(pInputPane, QStringLiteral("Calculation Input"));
    tabs->addTab(pChannelPane, QStringLiteral("Color Channels"));

    auto content = new QSplitter(nullptr);
    content->addWidget(tabs);
    content->addWidget(pTextureView);
    content->setStretchFactor(0, 1);
    content->setStretchFactor(1, 2);

    uMainWindow->setCentralWidget(content);
    uMainWindow->setStatusBar(pStatusBar);

    // clang-format off
    connect(
        pInputPane, &InputPane::signalAddTask,
        uRenderingManager.get(), &RenderingManager::slotAddTask,
        Qt::QueuedConnection
    );
    connect(
        uRenderingManager.get(), &RenderingManager::signalSceneRendered,
        pTextureView, &TextureView::slotSceneRendered,
        Qt::QueuedConnection
    );
    connect(
        pTextureView, &TextureView::signalUploadUnif,
        pChannelPane, &ChannelPane::slotUploadUnif,
        Qt::DirectConnection
    );
    connect(
        pChannelPane, &ChannelPane::signalUpdateGraphics,
        pTextureView, static_cast<void (QWidget::*)()>(&QWidget::update),
        Qt::DirectConnection
    );
    connect(
        pTextureView, &TextureView::signalGLContextInitialized,
        this, &Controller::slotGLContextInitialized,
        Qt::DirectConnection
    );

    // status messages
    connect(
        uRenderingManager.get(), &RenderingManager::signalStatusTemp,
        pStatusBar, &QStatusBar::showMessage,
        Qt::QueuedConnection
    );
    connect(
        pInputPane, &InputPane::signalStatusTemp,
        pStatusBar, &QStatusBar::showMessage,
        Qt::QueuedConnection
    );
    // clang-format on
}

void Controller::start()
{
    uMainWindow->show();
    uMainWindow->resize(AppConf::DEFAULT_MAINWINDOW_SIZE);
}

void Controller::slotGLContextInitialized(QOpenGLContext* context)
{
    uExportingManager->initialize(context);
    connect(pInputPane, &InputPane::signalExportImage, this, &Controller::slotExportImage, Qt::DirectConnection);
}

void Controller::slotExportImage()
{
    auto scene = pTextureView->scene();
    if (!scene) {
        return;
    }
    auto args = pChannelPane->channelArgs();

    auto filename = QFileDialog::getSaveFileName(
        uMainWindow.get(),
        QStringLiteral("Export Image"),
        QString(),
        QStringLiteral("Images (*.png)")
    );
    if (!filename.isEmpty()) {
        auto file = new QFile(filename);
        if (file->open(QIODeviceBase::WriteOnly | QIODeviceBase::Truncate)) {
            uExportingManager->requestExporting(scene, args, static_cast<QIODevice*>(file));
        }
    }
}
