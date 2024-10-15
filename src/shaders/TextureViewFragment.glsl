#version 400 core

uniform sampler2D points;

uniform float normFactor;
uniform float normRange;
uniform vec3 splineY[7];
uniform vec3 splineK[7];

in vec2 coord;
out vec4 color;

void main()
{
    float u = texture(points, coord).r;
    if (u >= 1.0) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float eval = abs(normRange) >= 0.00001 ? (exp(normFactor * u) - 1) / normRange : (normRange * u * (u - 1.0) * 0.5 + u);
    float t = clamp(6.0 * eval, 0.0, 5.9999995);
    int i = int(t);
    t -= i;
    float e = 1.0 - t;
    float t2 = t * t;
    float t3 = t2 * t;

    vec3 y0 = vec3(splineY[i]);
    vec3 y1 = vec3(splineY[i + 1]);
    vec3 s = y1 - y0;
    vec3 a = vec3(splineK[i]) - s;
    vec3 b = s - vec3(splineK[i + 1]);

    color = vec4(clamp(e * y0 + t * y1 + e * t * (e * a + t * b), 0.0, 1.0), 1.0);
}
