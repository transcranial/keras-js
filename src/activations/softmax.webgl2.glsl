#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D x;
out vec4 outColor;

void main() {
  vec2 size = textureSize(x, 0);
  int i = int(outTex.x * float(size[0]));

  float maxval = 0.0;
  for (int j = 0; j < size[1]; ++j) {
    maxval = max(maxval, texelFetch(x, ivec2(i, j), 0));
  }
  float sum = 0.0;
  for (int j = 0; j < size[1]; ++j) {
    sum += exp(texelFetch(x, ivec2(i, j), 0) - maxval);
  }

  outColor = exp(texelFetch(x, ivec2(i, j), 0) - maxval) / sum;
}
