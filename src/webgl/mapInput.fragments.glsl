#version 300 es
precision highp float;
precision highp isampler2D;

in vec2 outTex;
uniform sampler2D x;
uniform isampler2D rowIndexMap;
uniform isampler2D colIndexMap;
uniform int fragmentCols;
out vec4 outColor;

void main() {
  ivec2 inputSize = textureSize(x, 0);
  ivec2 outputSize = textureSize(rowIndexMap, 0);
  int out_x = int(float(outputSize[0]) * outTex.x);
  int out_y = int(float(outputSize[1]) * outTex.y);

  int rowIndex = texelFetch(rowIndexMap, ivec2(out_x, out_y), 0).r;
  int colIndex = texelFetch(colIndexMap, ivec2(out_x, out_y), 0).r;
  int fragmentIndex = int(floor(float(rowIndex) / float(inputSize[1])));

  if (rowIndex != -1 && colIndex != -1) {
    rowIndex = int(mod(float(rowIndex), float(inputSize[1])));
    colIndex = fragmentIndex * fragmentCols + colIndex;
    float val = texelFetch(x, ivec2(colIndex, rowIndex), 0).r;
    outColor = vec4(val);
  } else {
    outColor = vec4(0.0);
  }
}
