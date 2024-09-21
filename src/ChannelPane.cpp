#include "ChannelPane.h"

#include <QGridLayout>
#include <QtMath>

namespace ProjConf
{
extern double const LOG_NORM_SLIDER_SCALE;
extern int const LOG_NORM_SLIDER_RANGE;
extern int const COLOR_RANGE_SIXTH;
extern QString const CHANNEL_PANE_STYLE;
}  // namespace ProjConf

static double const COLOR_NORM = 1.0 / (ProjConf::COLOR_RANGE_SIXTH * 6);

ChannelPane::ChannelPane():
    QSplitter(Qt::Vertical, nullptr),
    sliderLogNorm(new QSlider(Qt::Horizontal, nullptr)),
    sliderChannelKnot(),
    bttnResetColor(new QPushButton(QStringLiteral("RESET"), nullptr)),
    pChannelCurves(new ChannelCurves()),
    logNormFactor(1.0),
    splineY(),
    splineK()
{
    setStyleSheet(ProjConf::CHANNEL_PANE_STYLE);

    sliderLogNorm->setRange(0, ProjConf::LOG_NORM_SLIDER_RANGE);
    connect(sliderLogNorm, &QSlider::valueChanged, [this](int value) {
        logNormFactor = qExp(ProjConf::LOG_NORM_SLIDER_SCALE * value / ProjConf::LOG_NORM_SLIDER_RANGE);
        emit signalUpdateGraphics();
    });

    for (size_t idx = 0; idx < 7; idx++) {
        for (size_t channel = 0; channel < 3; channel++) {
            auto slider = new QSlider(nullptr);
            slider->setProperty("channel", channel);
            slider->setRange(0, ProjConf::COLOR_RANGE_SIXTH * 6);
            connect(slider, &QSlider::valueChanged, [this, channel, idx](int value) {
                splineY[idx][channel] = static_cast<double>(value) * COLOR_NORM;
                updateSplineK(channel);
                emit signalUpdateGraphics();
            });
            sliderChannelKnot[idx][channel] = slider;
        }
    }

    setupLayout();

    // clang-format off
    connect(
        bttnResetColor, &QPushButton::clicked,
        this, &ChannelPane::resetColorSliders,
        Qt::DirectConnection
    );
    connect(
        pChannelCurves, &ChannelCurves::signalUploadUnif,
        this, &ChannelPane::slotUploadUnif,
        Qt::DirectConnection
    );
    connect(
        this, &ChannelPane::signalUpdateGraphics,
        [this]() { pChannelCurves->update(); }
    );
    // clang-format on
    resetColorSliders();
}

void ChannelPane::slotUploadUnif(QOpenGLShaderProgram* shader, int unifLogF, int unifSplineY, int unifSplineK)
{
    shader->setUniformValue(unifLogF, static_cast<float>(logNormFactor));
    shader->setUniformValueArray(unifSplineY, splineY.data(), 7);
    shader->setUniformValueArray(unifSplineK, splineK.data(), 7);
}

void ChannelPane::updateSplineK(size_t channel)
{
    // cubic spline interpolation under natural boundary condition
    constexpr double inv390 = 1.0 / 390;
    constexpr double inv780 = 1.0 / 780;
    auto const& y0 = splineY[0][channel];
    auto const& y1 = splineY[1][channel];
    auto const& y2 = splineY[2][channel];
    auto const& y3 = splineY[3][channel];
    auto const& y4 = splineY[4][channel];
    auto const& y5 = splineY[5][channel];
    auto const& y6 = splineY[6][channel];
    // clang-format off
    splineK[0][channel] = inv780 * (-989 * y0 + 1254 * y1 - 336 * y2 +  90 * y3 -  24 * y4 +    6 * y5 -       y6);
    splineK[1][channel] = inv390 * (-181 * y0 -   84 * y1 + 336 * y2 -  90 * y3 +  24 * y4 -    6 * y5 +       y6);
    splineK[2][channel] = inv780 * ( -97 * y0 -  582 * y1 -  12 * y2 + 630 * y3 - 168 * y4 +   42 * y5 -   7 * y6);
    splineK[3][channel] = inv390 * ( -13 * y0 +   78 * y1 - 312 * y2            + 312 * y4 -   78 * y5 +  13 * y6);
    splineK[4][channel] = inv780 * (   7 * y0 -   42 * y1 + 168 * y2 - 630 * y3 +  12 * y4 +  582 * y5 -  97 * y6);
    splineK[5][channel] = inv390 * (      -y0 +    6 * y1 -  24 * y2 +  90 * y3 - 336 * y4 +   84 * y5 + 181 * y6);
    splineK[6][channel] = inv780 * (       y0 -    6 * y1 +  24 * y2 -  90 * y3 + 336 * y4 - 1254 * y5 + 989 * y6);
    // clang-format on
}

void ChannelPane::resetColorSliders()
{
    sliderLogNorm->blockSignals(true);
    sliderLogNorm->setValue(0);
    sliderLogNorm->blockSignals(false);
    logNormFactor = 1.0;
    for (size_t idx = 0; idx < 7; idx++) {
        for (size_t channel = 0; channel < 3; channel++) {
            auto slider = sliderChannelKnot[idx][channel];
            auto value = idx * ProjConf::COLOR_RANGE_SIXTH;
            slider->blockSignals(true);
            slider->setValue(value);
            slider->blockSignals(false);
            splineY[idx][channel] = static_cast<double>(value) * COLOR_NORM;
        }
    }
    updateSplineK(0);
    updateSplineK(1);
    updateSplineK(2);
    emit signalUpdateGraphics();
}

void ChannelPane::setupLayout()
{
    auto tuners = new QWidget;
    auto grid = new QGridLayout(tuners);
    grid->setContentsMargins(8, 16, 8, 0);

    grid->addWidget(sliderLogNorm, 0, 0, 1, 7);

    auto addSliderRow = [this, grid](int row, size_t channel) {
        for (int idx = 0; idx < 7; idx++) {
            grid->addWidget(sliderChannelKnot[idx][channel], row, idx);
        }
    };
    addSliderRow(1, 0);
    addSliderRow(2, 1);
    addSliderRow(3, 2);
    grid->addWidget(bttnResetColor, 4, 0, 1, 7);

    addWidget(tuners);
    addWidget(pChannelCurves);
}
