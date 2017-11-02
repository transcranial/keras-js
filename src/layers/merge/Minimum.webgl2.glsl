#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D input1;
uniform sampler2D input2;
uniform int rows;
uniform int cols;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);

  float input1_val = texelFetch(input1, ivec2(out_x, out_y), 0).r;
  float input2_val = texelFetch(input2, ivec2(out_x, out_y), 0).r;

  outColor = vec4(min(input1_val, input2_val));
}
