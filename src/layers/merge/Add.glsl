#version 300 es
precision highp float;

in vec2 outTex;
uniform sampler2D input1;
uniform sampler2D input2;
out vec4 outColor;

void main() {
  ivec2 size = textureSize(input1, 0);
  int out_x = int(float(size[0]) * outTex.x);
  int out_y = int(float(size[1]) * outTex.y);

  float input1_val = texelFetch(input1, ivec2(out_x, out_y), 0).r;
  float input2_val = texelFetch(input2, ivec2(out_x, out_y), 0).r;

  outColor = vec4(input1_val + input2_val);
}
