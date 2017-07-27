#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D runningOutput;
uniform sampler2D input1;
uniform int rows;
uniform int cols;
uniform int concatAxis;
uniform int offsetStart;
uniform int offsetEnd;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);

  if (concatAxis == 0 && out_y >= offsetStart && out_y < offsetEnd) {
    outColor = vec4(texelFetch(input1, ivec2(out_x, out_y - offsetStart), 0).r);
  } else if (concatAxis == 1 && out_x >= offsetStart && out_x < offsetEnd) {
    outColor = vec4(texelFetch(input1, ivec2(out_x - offsetStart, out_y), 0).r);
  } else {
    outColor = texture(runningOutput, vec2(outTex.x, outTex.y));
  }
}
