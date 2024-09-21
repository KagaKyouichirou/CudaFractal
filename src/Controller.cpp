#include "Controller.h"

#include <QDockWidget>
#include <QTabWidget>

namespace ProjConf
{
extern QSize const DEFAULT_MAINWINDOW_SIZE;
}  // namespace ProjConf

Controller::Controller():
    QObject(nullptr),
    pMainWindow(std::make_unique<QMainWindow>(nullptr, Qt::Window)),
    pInputPane(new InputPane()),
    pChannelPane(new ChannelPane()),
    pTextureView(new TextureView()),
    uTaskManager(std::make_unique<TaskManager>())
{

    auto tabs = new QTabWidget(nullptr);
    tabs->addTab(pInputPane, QStringLiteral("Calculation Input"));
    tabs->addTab(pChannelPane, QStringLiteral("Color Channels"));

    auto pane = new QDockWidget(QStringLiteral("Controlling Pane"), nullptr);
    pane->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    pane->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    pane->setFloating(false);
    pane->setWidget(tabs);

    pMainWindow->setCentralWidget(pTextureView);
    pMainWindow->addDockWidget(Qt::LeftDockWidgetArea, pane);
    pMainWindow->setAnimated(false);

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
    pMainWindow->show();
    pMainWindow->resize(ProjConf::DEFAULT_MAINWINDOW_SIZE);
}
