#include "TextureScene.h"

TextureScene::TextureScene(std::unique_ptr<QOpenGLTexture> rendered):
    texture(std::move(rendered)), tX(0.0), tY(0.0), s(1.0)
{}

int TextureScene::width() const
{
    return texture->width();
}

int TextureScene::height() const
{
    return texture->height();
}

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

void TextureScene::bindTexture()
{
    texture->bind();
}

void TextureScene::releaseTexture()
{
    texture->release();
}
