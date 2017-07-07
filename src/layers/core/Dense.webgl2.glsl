#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D x;
uniform sampler2D kernel;
uniform sampler2D bias;
uniform bool use_bias;
uniform int M; // rows of kernel
uniform int N; // cols of kernel
out vec4 outColor;

void main() {
  int out_x = int(float(N) * outTex.x);

  float sum = 0.;
  for (int i = 0; i < M; ++i) {
    float a = texelFetch(x, ivec2(i, 0), 0).r;
    float b = texelFetch(kernel, ivec2(out_x, i), 0).r;
    sum += a * b;
  }

  if (use_bias) {
    sum += texelFetch(bias, ivec2(out_x, 0), 0).r;
    outColor = vec4(sum);
  } else {
    outColor = vec4(sum);
  }
}
