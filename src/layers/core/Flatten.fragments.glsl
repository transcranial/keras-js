#version 300 es
precision highp float;
precision highp sampler2DArray;

in vec2 outTex;
uniform sampler2DArray x;
uniform int outputSize;
uniform int inputRows;
uniform int inputCols;
out vec4 outColor;

void main() {
  int out_x = int(float(outputSize) * outTex.x);
  int out_y = 0;

  int i = int(mod(floor(float(out_x) / float(inputCols)), float(inputRows)));
  int j = int(mod(float(out_x), float(inputCols)));
  int k = int(floor(float(out_x) / (float(inputRows) * float(inputCols))));
  outColor = vec4(texelFetch(x, ivec3(j, i, k), 0).r);
}
