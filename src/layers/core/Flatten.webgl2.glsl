#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D x;
out vec4 outColor;

void main() {
  ivec2 size = textureSize(x, 0);
  int out_x = int(float(size[0]) * float(size[1]) * outTex.x);
  int out_y = 0;

  int i = int(mod(float(out_x), float(size[0])));
  int j = int(floor(float(out_x) / float(size[0])));
  outColor = vec4(texelFetch(x, ivec2(i, j), 0).r);
}
