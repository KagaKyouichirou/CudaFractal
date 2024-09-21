#version 400 core

uniform float logNormFactor;
uniform vec3 splineY[7];
uniform vec3 splineK[7];

uniform float strokePeak;
uniform float strokeNorm;
uniform float aspectRatio;

in vec2 textureCoord;
out vec4 color;

float logNormDeno;
float logNormFactorOverDenoMulAspect;
float reciprocalOfDerivativeAtX0;

float logNorm(float xScaled) {
    return log(1.0 + xScaled) / logNormDeno;
}

float logNormDerivative(float xScaled) {
    return logNormFactorOverDenoMulAspect / (1.0 + xScaled);
}

float funcOverDerivative(float x) {
    float xScaled = logNormFactor * x;
    float derivative = logNormDerivative(xScaled);
    float y = logNorm(xScaled);
    return ((y - textureCoord.y) * derivative + (x - textureCoord.x)) / (derivative * derivative + 2.0 - derivative * reciprocalOfDerivativeAtX0);
}

float newton(float x) {
    return x - funcOverDerivative(x);
}

void main()
{
    double t = 6.0 * double(textureCoord.x);
    int i = t == 6.0 ? 5 : int(t);
    t -= i;
    
    double e = 1.0 - t;
    double t2 = t * t;
    double t3 = t2 * t;

    dvec3 y0 = dvec3(splineY[i]);
    dvec3 y1 = dvec3(splineY[i + 1]);
    dvec3 s = y1 - y0;
    dvec3 a = dvec3(splineK[i]) - s;
    dvec3 b = s - dvec3(splineK[i + 1]);

    // interpolated values
    dvec3 f = e * y0 + t * y1 + e * t * (e * a + t * b);
    // 1st derivative
    dvec3 q = s + (e - t) * (e * a + t * b) + e * t * (b - a);
    q *= 6 * aspectRatio;
    // distance to tangent line and normalized
    dvec3 h = abs(f - textureCoord.y) * strokeNorm / sqrt(1.0 + q * q);

    dvec3 rgb = clamp((1.0 - 4 * h * h) * strokePeak, 0.0, 1.0);

    double l = length(rgb);
    if (l > 1.0) {
        rgb /= l;
    }

    if (l > 0.0) {
        color = vec4(rgb, 1.0);
        return;
    }

    // draw y = ln(1 + Wx) / (1 + W)
    
    // newton method to locate the ideal touch point
    logNormDeno = log(1.0 + logNormFactor);
    logNormFactorOverDenoMulAspect = aspectRatio * logNormFactor / logNormDeno;
    reciprocalOfDerivativeAtX0 = 1.0 / logNormDerivative(logNormFactor * textureCoord.x);

    float x = newton(textureCoord.x);
    x = newton(x);
    x = newton(x);
    float y = logNorm(logNormFactor * x);

    // square of distance
    float dx = x - textureCoord.x;
    float dy = y - textureCoord.y;
    float h2 = dx * dx + dy * dy;

    float res = clamp((1.0 - 4 * h2 * strokeNorm * strokeNorm) * strokePeak, 0.0, 1.0);
    color = vec4(res, res, res, 1.0);
}
