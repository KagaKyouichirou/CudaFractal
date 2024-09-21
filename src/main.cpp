#include <QApplication>

#include "Controller.h"

int main(int argc, char* argv[])
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication app(argc, argv);

    Controller c;
    c.start();

    return app.exec();
}
