#include "TextureScene.h"

TextureScene::TextureScene(QOpenGLTexture::Target target): QOpenGLTexture(target), tX(0.0), tY(0.0), s(1.0) {}

double TextureScene::translateX() const
{
    return tX;
}

double TextureScene::translateY() const
{
    return tY;
}

double TextureScene::scale() const
{
    return s;
}

void TextureScene::setTranslateX(double tX)
{
    this->tX = tX;
}

void TextureScene::setTranslateY(double tY)
{
    this->tY = tY;
}

void TextureScene::setScale(double s)
{
    this->s = s;
}
