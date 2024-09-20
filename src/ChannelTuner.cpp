#include "ChannelTuner.h"

ChannelTuner::ChannelTuner(QObject* parent): QObject(parent) {}

void ChannelTuner::slotUploadColorMatrices(QOpenGLShaderProgram* shader, int unifR, int unifG, int unifB)
{
    shader->setUniformValue(unifR, matrixColor[0]);
    shader->setUniformValue(unifG, matrixColor[1]);
    shader->setUniformValue(unifB, matrixColor[2]);
}

void ChannelTuner::slotColorTuned(size_t channel, size_t idx, double f0, double f1, double d0, double d1)
{
    // cubic Hermit interpolation for [0, 1]
    matrixColor[channel].setColumn(idx, QVector4D(f0, d0, 3 * f1 - d1 - 3 * f0 - 2 * d0, 2 * f0 - 2 * f1 + d0 + d1));
    //qDebug() << channel << matrixColor[channel];
    emit signalUpdate();
}