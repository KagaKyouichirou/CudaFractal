#version 400 core

uniform float normFactor;
uniform float normRange;
uniform vec3 splineY[7];
uniform vec3 splineK[7];

uniform float strokePeak;
uniform float strokeNorm;
uniform float aspectRatio;

in vec2 coord;
out vec4 color;

void main()
{
    float t = 6.0 * coord.x;
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
    f = abs(f - coord.y) * aspectRatio * strokeNorm;
    vec3 d2 = f * f;
    // slope
    vec3 q = s + (e - t) * (e * a + t * b) + e * t * (b - a);
    q *= 6.0 * aspectRatio;
    // distance to tangent line; normalized
    d2 /= (1.0 + q * q);

    vec3 rgb = clamp((1.0 - 4 * d2) * strokePeak, 0.0, 1.0);

    // draw y = [(e^(px) - 1)] / [e^p - 1] where p is normFactor and e^p - 1 is pre-computed as normRange
    float eval;
    float slope;
    if (abs(normRange) >= 0.00001) {
        float core = exp(normFactor * coord.x);
        eval = (core - 1.0) / normRange;
        slope = normFactor * core / normRange;
    } else {
        // degenerates to y = x + (e^p - 1) * x(x - 1) / 2
        eval = normRange * coord.x * (coord.x - 1) * 0.5 + coord.x;
        slope = normRange * (coord.x - 0.5) + 1.0;
    }
    float diff = abs(eval - coord.y) * aspectRatio * strokeNorm;
    slope *= aspectRatio;
    float dist2 = diff * diff / (1.0 + slope * slope);
    float res = clamp((1.0 - 4 * dist2) * strokePeak, 0.0, 1.0);
    rgb += vec3(res, res, res);

    float l = length(rgb);
    if (l > 1.0) {
        rgb /= l;
    }

    color = vec4(rgb, 1.0);
}
