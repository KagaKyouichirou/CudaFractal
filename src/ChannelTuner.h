#pragma once

#include <QObject>
#include <QOpenGLShaderProgram>
#

class ChannelTuner: public QObject
{
    Q_OBJECT
public:
    explicit ChannelTuner(QObject* parent);

public:
    void slotUploadColorMatrices(QOpenGLShaderProgram* shader, int unifR, int unifG, int unifB);

    void slotColorTuned(size_t channel, size_t idx, double f0, double f1, double d0, double d1);

signals:
    void signalUpdate();

private:
    std::array<QMatrix4x4, 3> matrixColor;
};