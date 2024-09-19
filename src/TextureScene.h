#pragma once

#include <QOpenGLBuffer>
#include <QOpenGLTexture>

class TextureScene
{
public:
    explicit TextureScene(std::unique_ptr<QOpenGLTexture> rendered);

public:
    int width() const;
    int height() const;

    double translateX() const;
    double translateY() const;
    double scale() const;

    void setTranslateX(double tX);
    void setTranslateY(double tY);
    void setScale(double s);

    void bindTexture();
    void releaseTexture();

public:
    TextureScene(TextureScene const&) = delete;
    TextureScene& operator=(TextureScene const&) = delete;
    TextureScene(TextureScene&&) = default;
    TextureScene& operator=(TextureScene&&) = default;

private:
    std::unique_ptr<QOpenGLTexture> texture;
    
    double tX;
    double tY;
    double s;
};
