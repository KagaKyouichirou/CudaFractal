#include "Controller.h"

#include "ChannelCurves.h"
#include "ChannelTuner.h"
#include "InputPane.h"
#include "TaskManager.h"
#include "TextureView.h"

#include <QGridLayout>

namespace ProjConf
{
extern QSize const DEFAULT_MAINWINDOW_SIZE;
}  // namespace ProjConf

Controller::Controller(): QObject(nullptr), pMainWindow(std::make_unique<QMainWindow>(nullptr, Qt::Window))
{
    auto central = new QWidget;

    auto pInputPane = new InputPane();
    auto pTextureView = new TextureView();
    auto pChannelCurves = new ChannelCurves();

    auto grid = new QGridLayout(central);
    grid->setColumnStretch(0, 2);
    grid->setColumnStretch(1, 7);
    grid->setRowStretch(0, 1);
    grid->setRowStretch(1, 1);
    grid->addWidget(pInputPane, 0, 0);
    grid->addWidget(pChannelCurves, 1, 0);
    grid->addWidget(pTextureView, 0, 1, 2, 1);

    pMainWindow->setCentralWidget(central);

    auto pTaskManager = new TaskManager(this);
    auto pChannelTuner = new ChannelTuner(this);
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
    connect(
        pInputPane, &InputPane::signalColorTuned,
        pChannelTuner, &ChannelTuner::slotColorTuned,
        Qt::DirectConnection
    );
    connect(
        pTextureView, &TextureView::signalUploadColorMatrices,
        pChannelTuner, &ChannelTuner::slotUploadColorMatrices,
        Qt::DirectConnection
    );
    connect(
        pChannelCurves, &ChannelCurves::signalUploadColorMatrices,
        pChannelTuner, &ChannelTuner::slotUploadColorMatrices,
        Qt::DirectConnection
    );
    connect(
        pChannelTuner, &ChannelTuner::signalUpdate,
        pTextureView, static_cast<void (QWidget::*)()>(&QWidget::update),
        Qt::DirectConnection
    );
    connect(
        pChannelTuner, &ChannelTuner::signalUpdate,
        pChannelCurves, static_cast<void (QWidget::*)()>(&QWidget::update),
        Qt::DirectConnection
    );
    // clang-format on
    connect(pInputPane, &InputPane::signalStatusTemp, [](QString hint) { qDebug() << hint; });

    pInputPane->resetColorSliders();
}

void Controller::start()
{
    pMainWindow->show();
    pMainWindow->resize(ProjConf::DEFAULT_MAINWINDOW_SIZE);
}
