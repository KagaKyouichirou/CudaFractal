#include "Controller.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    Controller c;
    c.start();

    return app.exec();
}
