#include <QApplication>

#include "Controller.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    Controller c;
    c.start();

    return app.exec();
}
