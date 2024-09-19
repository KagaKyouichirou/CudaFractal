#pragma once

#include "TaskArgs.h"

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>

class InputPane: public QScrollArea
{
    Q_OBJECT

public:
    explicit InputPane();

signals:
    void signalAddTask(TaskArgs task);
    void signalStatusTemp(QString hint);

protected:
    void resizeEvent(QResizeEvent* event) override final;

private:
    void render();

private:
    QComboBox* inputDimOption;
    QLineEdit* inputCenterX;
    QLineEdit* inputCenterY;
    QLineEdit* inputHalfUnit;
    QLineEdit* inputIterLimit;

    QPushButton* bttnRender;
};
