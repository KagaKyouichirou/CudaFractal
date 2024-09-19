#include "Controller.h"

#include "InputPane.h"
#include "TaskManager.h"
#include "TextureView.h"

#include <QHBoxLayout>

namespace ProjConf
{
extern QSize const DEFAULT_MAINWINDOW_SIZE;
}  // namespace ProjConf

Controller::Controller(): QObject(nullptr), pMainWindow(std::make_unique<QMainWindow>(nullptr, Qt::Window))
{
    auto central = new QWidget;

    auto pInputPane = new InputPane();
    auto pTextureView = new TextureView();

    auto hbox = new QHBoxLayout(central);
    hbox->setContentsMargins(0, 0, 0, 0);
    hbox->addWidget(pInputPane, 2);
    hbox->addWidget(pTextureView, 7);

    pMainWindow->setCentralWidget(central);

    auto pTaskManager = new TaskManager(this);
    // clang-format off
    connect(
        pInputPane, &InputPane::signalAddTask,
        pTaskManager, &TaskManager::slotAddTask,
        Qt::QueuedConnection
    );
    connect(
        pTaskManager, &TaskManager::signalSceneRendered,
        pTextureView, &TextureView::slotSceneRendered,
        Qt::QueuedConnection
    );
    // clang-format on
    connect(pInputPane, &InputPane::signalStatusTemp, [](QString hint) { qDebug() << hint; });
}

void Controller::start()
{
    pMainWindow->show();
    pMainWindow->resize(ProjConf::DEFAULT_MAINWINDOW_SIZE);
}
