#pragma once

#include <array>
#include <QVector3D>

struct ChannelArgs
{
    double normFactor;
    double normRange;

    std::array<QVector3D, 7> splineY;
    std::array<QVector3D, 7> splineK;

    explicit ChannelArgs(): normFactor(0.0), normRange(0.0), splineY(), splineK() {}
};