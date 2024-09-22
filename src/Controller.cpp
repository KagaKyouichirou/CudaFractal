#include "Controller.h"

#include <QSplitter>
#include <QTabWidget>

namespace ProjConf
{
extern QSize const DEFAULT_MAINWINDOW_SIZE;
}  // namespace ProjConf

Controller::Controller():
    QObject(nullptr),
    uMainWindow(std::make_unique<QMainWindow>(nullptr, Qt::Window)),
    pInputPane(new InputPane()),
    pChannelPane(new ChannelPane()),
    pTextureView(new TextureView()),
    uTaskManager(std::make_unique<TaskManager>())
{
    auto tabs = new QTabWidget(nullptr);
    tabs->addTab(pInputPane, QStringLiteral("Calculation Input"));
    tabs->addTab(pChannelPane, QStringLiteral("Color Channels"));

    auto content = new QSplitter(nullptr);
    content->addWidget(tabs);
    content->addWidget(pTextureView);

    uMainWindow->setCentralWidget(content);

    // clang-format off
    connect(
        pInputPane, &InputPane::signalAddTask,
        uTaskManager.get(), &TaskManager::slotAddTask,
        Qt::QueuedConnection
    );
    connect(
        uTaskManager.get(), &TaskManager::signalSceneRendered,
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
    // clang-format on
    connect(pInputPane, &InputPane::signalStatusTemp, [](QString hint) { qDebug() << hint; });
}

void Controller::start()
{
    uMainWindow->show();
    uMainWindow->resize(ProjConf::DEFAULT_MAINWINDOW_SIZE);
}
