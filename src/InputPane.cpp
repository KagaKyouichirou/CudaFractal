#include "InputPane.h"

#include <vector_types.h>
#include <QDoubleValidator>
#include <QGridLayout>
#include <QIntValidator>
#include <QLabel>

namespace ProjConf
{
extern QString const INPUT_PANE_STYLE;
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS;
extern QString const DEFAULT_CENTER_X;
extern QString const DEFAULT_CENTER_Y;
extern QString const DEFAULT_HALF_UNIT;
extern QString const DEFAULT_ITER_LIMIT;
}  // namespace ProjConf

InputPane::InputPane():
    QWidget(nullptr),
    inputDimOption(new QComboBox(nullptr)),
    inputCenterX(new QLineEdit(nullptr)),
    inputCenterY(new QLineEdit(nullptr)),
    inputHalfUnit(new QLineEdit(nullptr)),
    inputIterLimit(new QLineEdit(nullptr)),
    bttnRender(new QPushButton(QStringLiteral("RENDER"), nullptr))
{
    setStyleSheet(ProjConf::INPUT_PANE_STYLE);

    setupLayout();

    for (auto p : ProjConf::DIMENSION_OPTIONS) {
        auto w = p.first.x * p.second.x;
        auto h = p.first.y * p.second.y;
        inputDimOption->addItem(QString::asprintf("%d × %d", w, h));
    }

    auto dv = new QDoubleValidator(this);
    inputCenterX->setValidator(dv);
    inputCenterY->setValidator(dv);
    inputHalfUnit->setValidator(dv);

    inputIterLimit->setValidator(new QIntValidator(1, 2147483647, this));

    connect(bttnRender, &QPushButton::clicked, this, &InputPane::render, Qt::DirectConnection);

    // fill in initial input values
    inputDimOption->setCurrentIndex(0);
    inputCenterX->setText(ProjConf::DEFAULT_CENTER_X);
    inputCenterY->setText(ProjConf::DEFAULT_CENTER_Y);
    inputHalfUnit->setText(ProjConf::DEFAULT_HALF_UNIT);
    inputIterLimit->setText(ProjConf::DEFAULT_ITER_LIMIT);
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

void InputPane::setupLayout()
{
    auto grid = new QGridLayout(this);
    grid->setContentsMargins(8, 16, 8, 0);

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

    grid->setRowStretch(6, 1);
}
