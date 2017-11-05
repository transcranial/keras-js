#version 300 es
precision highp float;
precision highp isampler2D;

in vec2 outTex;
uniform sampler2D x;
uniform isampler2D rowIndexMap;
uniform isampler2D colIndexMap;
out vec4 outColor;

void main() {
  ivec2 size = textureSize(rowIndexMap, 0);
  int out_x = int(float(size[0]) * outTex.x);
  int out_y = int(float(size[1]) * outTex.y);

  int rowIndex = texelFetch(rowIndexMap, ivec2(out_x, out_y), 0).r;
  int colIndex = texelFetch(colIndexMap, ivec2(out_x, out_y), 0).r;

  if (rowIndex != -1 && colIndex != -1) {
    float val = texelFetch(x, ivec2(colIndex, rowIndex), 0).r;
    outColor = vec4(val);
  } else {
    outColor = vec4(0.0);
  }
}
