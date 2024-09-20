#pragma once

#include <QObject>
#include <QOpenGLShaderProgram>

class ChannelTuner: public QObject
{
    Q_OBJECT
public:
    explicit ChannelTuner(QObject* parent);

public:
    void slotUploadSplines(QOpenGLShaderProgram* shader, int unifSplineY, int unifSplineK);

    void slotSetSplineY(size_t channel, size_t idx, double y);
    void slotUpdateSplineK(size_t channel);

signals:
    void signalUpdateGL();

private:
    std::array<QVector3D, 7> splineY;
    std::array<QVector3D, 7> splineK;
};