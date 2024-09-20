#pragma once

#include "TaskArgs.h"

#include <QComboBox>
#include <QLineEdit>
#include <QMatrix4x4>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSlider>

class InputPane: public QScrollArea
{
    Q_OBJECT

public:
    explicit InputPane();

    void resetColorSliders();

signals:
    void signalAddTask(TaskArgs task);
    void signalStatusTemp(QString hint);

    void signalSetSplineY(size_t channel, size_t idx, double y);
    void signalUpdateSplineK(size_t channel);

protected:
    void resizeEvent(QResizeEvent* event) override final;

private:
    void render();
    void setupLayout();
    void tuneColor(size_t channel);

private:
    QComboBox* inputDimOption;
    QLineEdit* inputCenterX;
    QLineEdit* inputCenterY;
    QLineEdit* inputHalfUnit;
    QLineEdit* inputIterLimit;
    QPushButton* bttnRender;

    std::array<std::array<QSlider*, 7>, 3> colorSampleValues;
    QPushButton* bttnResetColor;
};
