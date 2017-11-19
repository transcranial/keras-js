#version 300 es
precision highp float;

in vec2 outTex;
uniform sampler2D runningOutput;
uniform sampler2D input1;
uniform int concatAxis;
uniform int offsetStart;
uniform int offsetEnd;
out vec4 outColor;

void main() {
  ivec2 size = textureSize(runningOutput, 0);
  int out_x = int(float(size[0]) * outTex.x);
  int out_y = int(float(size[1]) * outTex.y);

  if (concatAxis == 0 && out_y >= offsetStart && out_y < offsetEnd) {
    outColor = vec4(texelFetch(input1, ivec2(out_x, out_y - offsetStart), 0).r);
  } else if (concatAxis == 1 && out_x >= offsetStart && out_x < offsetEnd) {
    outColor = vec4(texelFetch(input1, ivec2(out_x - offsetStart, out_y), 0).r);
  } else {
    outColor = texture(runningOutput, vec2(outTex.x, outTex.y));
  }
}
