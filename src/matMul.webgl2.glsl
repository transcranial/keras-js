#version 300 es
precision highp float;

in vec2 outTex;
uniform sampler2D A;
uniform sampler2D B;
uniform sampler2D C;
uniform bool addC;
uniform int M; // rows of A
uniform int K; // common dimension
uniform int N; // cols of B
out vec4 outColor;

void main() {
  int out_x = int(float(N) * outTex.x);
  int out_y = int(float(M) * outTex.y);

  float sum = 0.;
  for (int i = 0; i < K; ++i) {
    float a = texelFetch(A, ivec2(i, out_y), 0).r;
    float b = texelFetch(B, ivec2(out_x, i), 0).r;
    sum += a * b;
  }

  if (addC) {
    sum += texelFetch(C, ivec2(out_x, 0), 0).r;
  }

  outColor = vec4(sum);
}
