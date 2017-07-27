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
}
