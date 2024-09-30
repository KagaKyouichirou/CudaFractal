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
    ~InputPane() override final = default;

signals:
    void signalAddTask(TaskArgs task);
    void signalExportImage();
    void signalStatusTemp(QString const& text, int timeout = 0);

    void signalSetSplineY(size_t channel, size_t idx, double y);
    void signalUpdateSplineK(size_t channel);

private:
    void render();

    void setupLayout();

private:
    // resolution
    QComboBox* comboDimOption;

    // Center.X
    QPushButton* signCenterX;
    QLineEdit* fracCenterX;
    QComboBox* expoCenterX;

    // Center.Y
    QPushButton* signCenterY;
    QLineEdit* fracCenterY;
    QComboBox* expoCenterY;

    // Half Unit
    QLineEdit* fracHalfUnit;
    QComboBox* expoHalfUnit;

    // Iter Limit
    QLineEdit* lineIterLimit;

    QPushButton* bttnRender;
    QPushButton* bttnExport;
};
