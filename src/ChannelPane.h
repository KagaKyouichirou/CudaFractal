#pragma once

#include <QOpenGLShaderProgram>
#include <QPushButton>
#include <QSlider>
#include <QSplitter>
#include "ChannelCurves.h"

class ChannelPane: public QSplitter
{
    Q_OBJECT

public:
    explicit ChannelPane();
    virtual ~ChannelPane() override final = default;

public:
    void slotUploadUnif(QOpenGLShaderProgram* sh, int unifLogF, int unifLogN, int unifSpY, int unifSpK);

signals:
    void signalUpdateGraphics();

private:
    void updateSplineK(size_t channel);
    void resetColorSliders();
    void setupLayout();

private:
    QSlider* sliderLogNorm;
    std::array<std::array<QSlider*, 3>, 7> sliderChannelKnot;
    QPushButton* bttnResetColor;
    ChannelCurves* pChannelCurves;

    double logFactor;
    double logNorm;

    std::array<QVector3D, 7> splineY;
    std::array<QVector3D, 7> splineK;
};