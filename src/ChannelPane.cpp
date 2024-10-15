#include "ChannelPane.h"

#include <QGridLayout>
#include <QtMath>

namespace AppConf
{
extern double const GREY_NORM_SLIDER_SCALE;
extern int const GREY_NORM_SLIDER_RANGE_HALF;
extern int const COLOR_RANGE_SIXTH;
extern QString const CHANNEL_PANE_STYLE;
extern int const COLOR_SLIDER_HALF_WIDTH;
}  // namespace AppConf

static double const COLOR_NORM = 1.0 / (AppConf::COLOR_RANGE_SIXTH * 6);

ChannelPane::ChannelPane():
    QSplitter(Qt::Vertical, nullptr),
    sliderLogCurve(new QSlider(Qt::Horizontal, nullptr)),
    sliderChannelKnot(),
    bttnResetColor(new QPushButton(QStringLiteral("RESET"), nullptr)),
    pChannelCurves(new ChannelCurves()),
    args()
{
    setStyleSheet(AppConf::CHANNEL_PANE_STYLE);
    sliderLogCurve->setRange(0, 2 * AppConf::GREY_NORM_SLIDER_RANGE_HALF);
    connect(sliderLogCurve, &QSlider::valueChanged, [this](int value) {
        args.normFactor = AppConf::GREY_NORM_SLIDER_SCALE * (static_cast<float>(value) / AppConf::GREY_NORM_SLIDER_RANGE_HALF - 1.0);
        args.normRange = qExp(args.normFactor) - 1;
        emit signalUpdateGraphics();
    });

    for (size_t idx = 0; idx < 7; idx++) {
        for (size_t channel = 0; channel < 3; channel++) {
            auto slider = new QSlider(nullptr);
            slider->setFixedWidth(2 * AppConf::COLOR_SLIDER_HALF_WIDTH);
            slider->setMinimumHeight(50);
            slider->setProperty("channel", channel);
            slider->setRange(0, AppConf::COLOR_RANGE_SIXTH * 6);
            connect(slider, &QSlider::valueChanged, [this, channel, idx](int value) {
                args.splineY[idx][channel] = static_cast<double>(value) * COLOR_NORM;
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

ChannelArgs* ChannelPane::channelArgs() const
{
    return new ChannelArgs(args);
}

void ChannelPane::slotUploadUnif(QOpenGLShaderProgram* sh, int unifNormF, int unifNormR, int unifSpY, int unifSpK)
{
    sh->setUniformValue(unifNormF, static_cast<float>(args.normFactor));
    sh->setUniformValue(unifNormR, static_cast<float>(args.normRange));
    sh->setUniformValueArray(unifSpY, args.splineY.data(), 7);
    sh->setUniformValueArray(unifSpK, args.splineK.data(), 7);
}

void ChannelPane::updateSplineK(size_t channel)
{
    // cubic spline interpolation under natural boundary condition
    constexpr double inv390 = 1.0 / 390;
    constexpr double inv780 = 1.0 / 780;
    auto const& y0 = args.splineY[0][channel];
    auto const& y1 = args.splineY[1][channel];
    auto const& y2 = args.splineY[2][channel];
    auto const& y3 = args.splineY[3][channel];
    auto const& y4 = args.splineY[4][channel];
    auto const& y5 = args.splineY[5][channel];
    auto const& y6 = args.splineY[6][channel];
    // clang-format off
    args.splineK[0][channel] = inv780 * (-989 * y0 + 1254 * y1 - 336 * y2 +  90 * y3 -  24 * y4 +    6 * y5 -       y6);
    args.splineK[1][channel] = inv390 * (-181 * y0 -   84 * y1 + 336 * y2 -  90 * y3 +  24 * y4 -    6 * y5 +       y6);
    args.splineK[2][channel] = inv780 * ( -97 * y0 -  582 * y1 -  12 * y2 + 630 * y3 - 168 * y4 +   42 * y5 -   7 * y6);
    args.splineK[3][channel] = inv390 * ( -13 * y0 +   78 * y1 - 312 * y2            + 312 * y4 -   78 * y5 +  13 * y6);
    args.splineK[4][channel] = inv780 * (   7 * y0 -   42 * y1 + 168 * y2 - 630 * y3 +  12 * y4 +  582 * y5 -  97 * y6);
    args.splineK[5][channel] = inv390 * (      -y0 +    6 * y1 -  24 * y2 +  90 * y3 - 336 * y4 +   84 * y5 + 181 * y6);
    args.splineK[6][channel] = inv780 * (       y0 -    6 * y1 +  24 * y2 -  90 * y3 + 336 * y4 - 1254 * y5 + 989 * y6);
    // clang-format on
}

void ChannelPane::resetColorSliders()
{
    sliderLogCurve->blockSignals(true);
    sliderLogCurve->setValue(AppConf::GREY_NORM_SLIDER_RANGE_HALF);
    args.normFactor = 0.0;
    args.normRange = 0.0;
    sliderLogCurve->blockSignals(false);
    for (size_t idx = 0; idx < 7; idx++) {
        for (size_t channel = 0; channel < 3; channel++) {
            auto slider = sliderChannelKnot[idx][channel];
            auto value = idx * AppConf::COLOR_RANGE_SIXTH;
            slider->blockSignals(true);
            slider->setValue(value);
            slider->blockSignals(false);
            args.splineY[idx][channel] = static_cast<double>(value) * COLOR_NORM;
        }
    }
    updateSplineK(0);
    updateSplineK(1);
    updateSplineK(2);
    emit signalUpdateGraphics();
}

void ChannelPane::setupLayout()
{
    auto tuners = new QWidget(nullptr);
    auto grid = new QGridLayout(tuners);
    constexpr int M = 8;
    grid->setContentsMargins(M, 8, M, 0);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(3, 1);
    grid->setColumnStretch(5, 1);
    grid->setColumnStretch(7, 1);
    grid->setColumnStretch(9, 1);
    grid->setColumnStretch(11, 1);
    grid->addWidget(sliderLogCurve, 0, 0, 1, 13);
    auto addSliderRow = [this, grid](int row, size_t channel) {
        for (int idx = 0; idx < 7; idx++) {
            grid->addWidget(sliderChannelKnot[idx][channel], row, idx * 2, Qt::AlignHCenter);
        }
    };
    addSliderRow(1, 0);
    grid->setRowStretch(1, 1);
    addSliderRow(2, 1);
    grid->setRowStretch(2, 1);
    addSliderRow(3, 2);
    grid->setRowStretch(3, 1);
    grid->addWidget(bttnResetColor, 4, 0, 1, 13);

    auto wrapper = new QWidget(nullptr);
    auto vbox = new QVBoxLayout(wrapper);
    auto m = M + AppConf::COLOR_SLIDER_HALF_WIDTH;
    vbox->setContentsMargins(m, 0, m, 0);
    vbox->addWidget(pChannelCurves);
    addWidget(tuners);
    addWidget(wrapper);
}
