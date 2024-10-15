#pragma once

#include "ChannelArgs.h"
#include "ChannelCurves.h"

#include <QOpenGLShaderProgram>
#include <QPushButton>
#include <QSlider>
#include <QSplitter>

class ChannelPane: public QSplitter
{
    Q_OBJECT

public:
    explicit ChannelPane();
    ~ChannelPane() override final = default;

    ChannelArgs* channelArgs() const;

public:
    void slotUploadUnif(QOpenGLShaderProgram* sh, int unifNormF, int unifNormR, int unifSpY, int unifSpK);

signals:
    void signalUpdateGraphics();

private:
    void updateSplineK(size_t channel);
    void resetColorSliders();
    void setupLayout();

private:
    QSlider* sliderLogCurve;
    std::array<std::array<QSlider*, 3>, 7> sliderChannelKnot;
    QPushButton* bttnResetColor;
    ChannelCurves* pChannelCurves;

    ChannelArgs args;
};