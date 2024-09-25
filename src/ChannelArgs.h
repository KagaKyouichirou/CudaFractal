#pragma once

#include <array>
#include <QVector3D>

struct ChannelArgs
{
    double logFactor;
    double logNorm;

    std::array<QVector3D, 7> splineY;
    std::array<QVector3D, 7> splineK;

    explicit ChannelArgs(): logFactor(0.0), logNorm(1.0), splineY(), splineK() {}
};