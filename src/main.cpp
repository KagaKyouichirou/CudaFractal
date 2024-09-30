#include "Controller.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    Controller c;
    c.start();

    return app.exec();

    // {
    //     FixedPoint8U30Naive half = FixedPoint8U30Naive::fromBytes(std::array<uint8_t, 30>{0, 0, 0, 0, 2}, false);
    //     auto oX = half;
    //     oX.mul(1023);
    //     oX.flip();
    //     auto oY = half;
    //     oY.mul(1023);
    //     oY.flip();
    //     auto step = half;
    //     step.dou();
    //     auto eX = step;
    //     eX.mul(260);
    //     eX.add(oX);
    //     auto eY = step;
    //     eY.mul(170);
    //     eY.add(oY);
    //     auto real = eX;
    //     auto imgn = eY;
    //     auto real2 = real;
    //     real2.sqr();
    //     auto imgn2 = imgn;
    //     imgn2.sqr();
    //     uint16_t k = 0;
    //     while (k < 2 && FixedPoint8U30Naive::checkNorm(real2, imgn2)) {
    //         imgn.mul(real);
    //         imgn.dou();
    //         imgn.add(eY);
    //         real = imgn2;
    //         real.flip();
    //         real.add(real2);
    //         real.add(eX);
    //         real2 = real;
    //         real2.sqr();
    //         imgn2 = imgn;
    //         imgn2.sqr();
    //         k++;
    //     }
    //     qDebug() << k << QByteArrayView(real.intoBytes().data(), 30);
    // }

}
