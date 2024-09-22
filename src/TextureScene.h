#pragma once

#include <QOpenGLTexture>

class TextureScene: public QOpenGLTexture
{
public:
    explicit TextureScene(QOpenGLTexture::Target target);
    ~TextureScene() = default;

public:
    double translateX() const;
    double translateY() const;
    double scale() const;

    void setTranslateX(double tX);
    void setTranslateY(double tY);
    void setScale(double s);

public:
    TextureScene(TextureScene const&) = delete;
    TextureScene& operator=(TextureScene const&) = delete;
    TextureScene(TextureScene&&) = delete;
    TextureScene& operator=(TextureScene&&) = delete;

private:
    double tX;
    double tY;
    double s;
};
