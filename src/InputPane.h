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
    void signalColorTuned(size_t channel, size_t idx, double f0, double f1, double d0, double d1);

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

    // [R, G, B]
    std::array<std::array<QSlider*, 5>, 3> colorSampleValues;
    std::array<std::array<QSlider*, 2>, 3> colorBoundaryFactors;
    QPushButton* bttnResetColor;
};
