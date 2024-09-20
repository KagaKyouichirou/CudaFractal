#version 400 core

uniform sampler2D points;

uniform vec3 splineY[7];
uniform vec3 splineK[7];

in vec2 coord;
out vec4 color;

void main()
{
    double t = 6 * double(texture(points, coord).r);
    if (t == 6.0) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    int i = int(t);
    t -= i;
    double e = 1.0 - t;
    double t2 = t * t;
    double t3 = t2 * t;

    dvec3 y0 = dvec3(splineY[i]);
    dvec3 y1 = dvec3(splineY[i + 1]);
    dvec3 s = y1 - y0;
    dvec3 a = dvec3(splineK[i]) - s;
    dvec3 b = s - dvec3(splineK[i + 1]);

    color = vec4(clamp(e * y0 + t * y1 + e * t * (e * a + t * b), 0.0, 1.0), 1.0);
}
