#include "InputPane.h"

#include <QDoubleValidator>
#include <QGridLayout>
#include <QIntValidator>
#include <QLabel>
#include <QScrollArea>
#include <QScrollBar>
#include <QtMath>

namespace ProjConf
{
extern QString const INPUT_PANE_STYLE;
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS;
extern QString const DEFAULT_CENTER_X;
extern QString const DEFAULT_CENTER_Y;
extern QString const DEFAULT_HALF_UNIT;
extern QString const DEFAULT_ITER_LIMIT;
extern int const COLOR_RANGE_SIXTH;
}  // namespace ProjConf

constexpr double HALF_PI_MINUS = 1.57075;

static double const COLOR_NORM = 1.0 / (ProjConf::COLOR_RANGE_SIXTH * 6);

InputPane::InputPane():
    QScrollArea(nullptr),
    inputDimOption(new QComboBox(nullptr)),
    inputCenterX(new QLineEdit(nullptr)),
    inputCenterY(new QLineEdit(nullptr)),
    inputHalfUnit(new QLineEdit(nullptr)),
    inputIterLimit(new QLineEdit(nullptr)),
    bttnRender(new QPushButton(QStringLiteral("RENDER"), nullptr)),
    colorSampleValues(),
    bttnResetColor(new QPushButton(QStringLiteral("RESET"), nullptr))
{
    horizontalScrollBar()->setEnabled(false);
    setStyleSheet(ProjConf::INPUT_PANE_STYLE);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto buildSlider = []() {
        auto slider = new QSlider(nullptr);
        slider->setRange(0, ProjConf::COLOR_RANGE_SIXTH * 6);
        return slider;
    };
    for (size_t channel = 0; channel < 3; channel++) {
        for (size_t idx = 0; idx < 7; idx++) {
            auto slider = buildSlider();
            connect(slider, &QSlider::valueChanged, [this, channel, idx](int value) {
                emit signalSetSplineY(channel, idx, static_cast<double>(value) * COLOR_NORM);
                emit signalUpdateSplineK(channel);
            });
            colorSampleValues[channel][idx] = slider;
        }
    }

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
    connect(bttnResetColor, &QPushButton::clicked, this, &InputPane::resetColorSliders, Qt::DirectConnection);

    // fill in initial input values
    inputDimOption->setCurrentIndex(0);
    inputCenterX->setText(ProjConf::DEFAULT_CENTER_X);
    inputCenterY->setText(ProjConf::DEFAULT_CENTER_Y);
    inputHalfUnit->setText(ProjConf::DEFAULT_HALF_UNIT);
    inputIterLimit->setText(ProjConf::DEFAULT_ITER_LIMIT);
}

void InputPane::resetColorSliders()
{
    for (size_t channel = 0; channel < 3; channel++) {
        for (size_t idx = 0; idx < 7; idx++) {
            auto value = idx * ProjConf::COLOR_RANGE_SIXTH;
            auto& slider = colorSampleValues[channel][idx];
            slider->blockSignals(true);
            slider->setValue(value);
            slider->blockSignals(false);
            emit signalSetSplineY(channel, idx, static_cast<double>(value) * COLOR_NORM);
        }
        emit signalUpdateSplineK(channel);
    }
}

void InputPane::resizeEvent(QResizeEvent* event)
{
    widget()->resize(width(), widget()->height());
    QScrollArea::resizeEvent(event);
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
    auto content = new QWidget;
    auto vbox = new QVBoxLayout(content);

    auto paneCuda = new QWidget;
    auto gridCuda = new QGridLayout(paneCuda);
    gridCuda->setContentsMargins(8, 16, 8, 0);

    gridCuda->addWidget(new QLabel(QStringLiteral("Resolution")), 0, 0);
    gridCuda->addWidget(inputDimOption, 0, 1);

    gridCuda->addWidget(new QLabel(QStringLiteral("Center.X")), 1, 0);
    gridCuda->addWidget(inputCenterX, 1, 1);

    gridCuda->addWidget(new QLabel(QStringLiteral("Center.Y")), 2, 0);
    gridCuda->addWidget(inputCenterY, 2, 1);

    gridCuda->addWidget(new QLabel(QStringLiteral("Half Unit")), 3, 0);
    gridCuda->addWidget(inputHalfUnit, 3, 1);

    gridCuda->addWidget(new QLabel(QStringLiteral("Iter Limit")), 4, 0);
    gridCuda->addWidget(inputIterLimit, 4, 1);

    gridCuda->addWidget(bttnRender, 5, 0, 1, 2);

    vbox->addWidget(paneCuda);

    auto paneColor = new QWidget;
    auto gridColor = new QGridLayout(paneColor);
    gridColor->setContentsMargins(8, 16, 8, 0);

    auto addSliderRow = [this, gridColor](int row, size_t channel) {
        for (int idx = 0; idx < 7; idx++) {
            gridColor->addWidget(colorSampleValues[channel][idx], row, idx);
        }
    };

    auto centeredLabel = [](QString const& text) {
        auto label = new QLabel(text, nullptr);
        label->setAlignment(Qt::AlignCenter);
        return label;
    };
    addSliderRow(0, 0);
    gridColor->addWidget(centeredLabel(QStringLiteral("Channel R")), 1, 0, 1, 7);
    addSliderRow(2, 1);
    gridColor->addWidget(centeredLabel(QStringLiteral("Channel G")), 3, 0, 1, 7);
    addSliderRow(4, 2);
    gridColor->addWidget(centeredLabel(QStringLiteral("Channel B")), 5, 0, 1, 7);

    gridColor->addWidget(bttnResetColor, 6, 0, 1, 7);

    vbox->addWidget(paneColor);

    setWidget(content);
    //setWidgetResizable(true);
}

constexpr double mean(double x, double y)
{
    return (x + y) / 2;
}

void InputPane::tuneColor(size_t channel) {}
