#include "InputPane.h"

#include <QDoubleValidator>
#include <QGridLayout>
#include <QIntValidator>
#include <QLabel>
#include <QScrollArea>
#include <QScrollBar>

namespace ProjConf
{
extern QString const INPUT_PANE_STYLE;
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS;
}  // namespace ProjConf

InputPane::InputPane():
    QScrollArea(nullptr),
    inputDimOption(new QComboBox(nullptr)),
    inputCenterX(new QLineEdit(nullptr)),
    inputCenterY(new QLineEdit(nullptr)),
    inputHalfUnit(new QLineEdit(nullptr)),
    inputIterLimit(new QLineEdit(nullptr)),
    bttnRender(new QPushButton(QStringLiteral("RENDER"), nullptr))
{
    horizontalScrollBar()->setEnabled(false);
    setStyleSheet(ProjConf::INPUT_PANE_STYLE);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    for (auto p : ProjConf::DIMENSION_OPTIONS) {
        auto w = p.first.x * p.second.x;
        auto h = p.first.y * p.second.y;
        inputDimOption->addItem(QString::asprintf("%d x %d", w, h));
    }

    auto content = new QWidget;
    auto grid = new QGridLayout(content);
    grid->setContentsMargins(16, 16, 16, 16);
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 1);
    grid->addWidget(new QLabel(QStringLiteral("Resolution")), 0, 0);
    grid->addWidget(inputDimOption, 0, 1);
    grid->addWidget(new QLabel(QStringLiteral("Center.X")), 1, 0);
    grid->addWidget(inputCenterX, 1, 1);
    grid->addWidget(new QLabel(QStringLiteral("Center.Y")), 2, 0);
    grid->addWidget(inputCenterY, 2, 1);
    grid->addWidget(new QLabel(QStringLiteral("Half Unit")), 3, 0);
    grid->addWidget(inputHalfUnit, 3, 1);
    grid->addWidget(new QLabel(QStringLiteral("Iter Limit")), 4, 0);
    grid->addWidget(inputIterLimit, 4, 1);
    grid->addWidget(bttnRender, 5, 0, 1, 2);

    setWidget(content);

    auto dv = new QDoubleValidator(this);
    inputCenterX->setValidator(dv);
    inputCenterY->setValidator(dv);
    inputHalfUnit->setValidator(dv);

    inputIterLimit->setValidator(new QIntValidator(1, 2147483647, this));

    connect(bttnRender, &QPushButton::clicked, this, &InputPane::render, Qt::DirectConnection);
}

void InputPane::resizeEvent(QResizeEvent* event)
{
    widget()->resize(width(), widget()->height());
}

void InputPane::render()
{
    bool ok = false;
    auto x = inputCenterX->text().toDouble(&ok);
    if (!(ok && x >= -2.0 && x <= 1.0)) {
        emit signalStatusTemp(QStringLiteral("Center.X should be within the range [-2.0, 1.0]"));
        return;
    }
    auto y = inputCenterY->text().toDouble(&ok);
    if (!(ok && y >= -1.5 && y <= 1.5)) {
        emit signalStatusTemp(QStringLiteral("Center.Y should be within the range [-1.5, 1.5]"));
        return;
    }
    auto h = inputHalfUnit->text().toDouble(&ok);
    if (!(ok && h > 0.0 && h <= 0.004)) {
        emit signalStatusTemp(QStringLiteral("Half-Unit should be within the range (0.0, 0.004]"));
        return;
    }
    auto l = inputIterLimit->text().toInt(&ok);
    if (!(ok && l >= 16 && l <= 65535)) {
        emit signalStatusTemp(QStringLiteral("Iter-Limit should be at least 16 and not exceeding 65535"));
    }
    auto [dGrid, dBlock] = ProjConf::DIMENSION_OPTIONS[inputDimOption->currentIndex()];
    emit signalAddTask(TaskArgs{dGrid, dBlock, x, y, h, static_cast<uint16_t>(l)});
}