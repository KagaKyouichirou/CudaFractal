#pragma once

#include <vector_types.h>
#include <QMainWindow>
#include <QObject>

class Controller: public QObject
{
    Q_OBJECT

public:
    explicit Controller();
    ~Controller() = default;

    void start();

private:
    std::unique_ptr<QMainWindow> pMainWindow;
};