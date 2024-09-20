#include "ChannelTuner.h"

ChannelTuner::ChannelTuner(QObject* parent): QObject(parent) {}

void ChannelTuner::slotUploadSplines(QOpenGLShaderProgram* shader, int unifSplineY, int unifSplineK)
{
    shader->setUniformValueArray(unifSplineY, splineY.data(), 7);
    shader->setUniformValueArray(unifSplineK, splineK.data(), 7);
}

void ChannelTuner::slotSetSplineY(size_t channel, size_t idx, double y)
{
    // set the value silently without updating splineK
    splineY[idx][channel] = y;
}

void ChannelTuner::slotUpdateSplineK(size_t channel)
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
    splineK[0][channel] = inv780 * (-989 * y0 + 1254 * y1 - 336 * y2 +  90 * y3 -  24 * y4 +    6 * y5 -       y6);
    splineK[1][channel] = inv390 * (-181 * y0 -   84 * y1 + 336 * y2 -  90 * y3 +  24 * y4 -    6 * y5 +       y6);
    splineK[2][channel] = inv780 * ( -97 * y0 -  582 * y1 -  12 * y2 + 630 * y3 - 168 * y4 +   42 * y5 -   7 * y6);
    splineK[3][channel] = inv390 * ( -13 * y0 +   78 * y1 - 312 * y2            + 312 * y4 -   78 * y5 +  13 * y6);
    splineK[4][channel] = inv780 * (   7 * y0 -   42 * y1 + 168 * y2 - 630 * y3 +  12 * y4 +  582 * y5 -  97 * y6);
    splineK[5][channel] = inv390 * (     - y0 +    6 * y1 -  24 * y2 +  90 * y3 - 336 * y4 +   84 * y5 + 181 * y6);
    splineK[6][channel] = inv780 * (       y0 -    6 * y1 +  24 * y2 -  90 * y3 + 336 * y4 - 1254 * y5 + 989 * y6);
    emit signalUpdateGL();
}
