#include <QApplication>

#include "OpenGLWidget.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    OpenGLWidget widget;
    widget.resize(800, 800);
    widget.show();

    return app.exec();
}
