#version 400 core

uniform float logFactor;
uniform float logNorm;
uniform vec3 splineY[7];
uniform vec3 splineK[7];

uniform float strokePeak;
uniform float strokeNorm;
uniform float aspectRatio;

in vec2 textureCoord;
out vec4 color;

void main()
{
    float t = 6.0 * textureCoord.x;
    int i = t >= 6.0 ? 5 : int(t);
    t -= i;
    
    float e = 1.0 - t;
    float t2 = t * t;
    float t3 = t2 * t;

    vec3 y0 = splineY[i];
    vec3 y1 = splineY[i + 1];
    vec3 s = y1 - y0;
    vec3 a = splineK[i] - s;
    vec3 b = s - splineK[i + 1];

    // interpolated values
    vec3 f = e * y0 + t * y1 + e * t * (e * a + t * b);
    f = abs(f - textureCoord.y) * aspectRatio * strokeNorm;
    vec3 d2 = f * f;
    // 1st derivative
    vec3 q = s + (e - t) * (e * a + t * b) + e * t * (b - a);
    q *= 6.0 * aspectRatio;
    // distance to tangent line and normalized
    d2 /= (1.0 + q * q);

    vec3 rgb = clamp((1.0 - 4 * d2) * strokePeak, 0.0, 1.0);

    // draw y = ln(1 + Wx) / (1 + W)
    float xScaled = logFactor * textureCoord.x;
    float logXScaledPlusOne = log(1.0 + xScaled);
    float key = aspectRatio * logNorm - 1.0 / logFactor;
    float diff;
    float slope;
    if (textureCoord.x >= 0.0 * key) {
        diff = abs(logXScaledPlusOne * logNorm - textureCoord.y) * aspectRatio;
        slope = logFactor * logNorm / (1.0 + xScaled) * aspectRatio;
    } else {
        // use x = (exp(y / aspectRatio * ln(1 + W)) - 1) / W instead
        float core = exp(logXScaledPlusOne * textureCoord.y / aspectRatio);
        diff = abs((core - 1.0) / logFactor - textureCoord.x);
        slope = core / (aspectRatio * logFactor * logNorm);
    }
    diff *= strokeNorm;
    float dist2 = diff * diff / (1.0 + slope * slope);
    float res = clamp((1.0 - 4 * dist2) * strokePeak, 0.0, 1.0);
    rgb += vec3(res, res, res);

    float l = length(rgb);
    if (l > 1.0) {
        rgb /= l;
    }

    color = vec4(rgb, 1.0);
}
