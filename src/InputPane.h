#pragma once

#include "TaskArgs.h"

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>

class InputPane: public QWidget
{
    Q_OBJECT

public:
    explicit InputPane();
    virtual ~InputPane() override final = default;

signals:
    void signalAddTask(TaskArgs task);
    void signalExportImage();
    void signalStatusTemp(QString hint);

    void signalSetSplineY(size_t channel, size_t idx, double y);
    void signalUpdateSplineK(size_t channel);

private:    
    void render();

    void setupLayout();

private:
    QComboBox* inputDimOption;
    QLineEdit* inputCenterX;
    QLineEdit* inputCenterY;
    QLineEdit* inputHalfUnit;
    QLineEdit* inputIterLimit;
    QPushButton* bttnRender;
    QPushButton* bttnExport;
};
