#version 400 core

uniform vec3 splineY[7];
uniform vec3 splineK[7];

uniform float strokePeak;
uniform float strokeNorm;

in vec2 textureCoord;
out vec4 color;

void main()
{
    double t = double(textureCoord.x);
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
    // distance to tangent and normalized
    dvec3 h = abs(f - textureCoord.y) * strokeNorm / sqrt(1.0 + q * q);

    dvec3 rgb = clamp((1.0 - 4 * h * h) * strokePeak, 0.0, 1.0);

    double l = length(rgb);
    if (l > 1.0) {
        rgb *= 1.0 / l;
    }

    color = vec4(rgb, 1.0);
}
