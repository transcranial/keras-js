#version 300 es
precision highp float;
precision highp isampler2D;

in vec2 outTex;
uniform sampler2D matMulResult;
uniform isampler2D rowIndexMap;
uniform isampler2D colIndexMap;
uniform sampler2D bias;
uniform bool use_bias;
uniform int inputFragmentCols;
uniform int outputFragmentCols;
out vec4 outColor;

void main() {
  int inputFragmentRows = textureSize(matMulResult, 0)[1];
  int outputFragmentRows = textureSize(rowIndexMap, 0)[1];
  int summationLength = textureSize(rowIndexMap, 0)[0];
  int out_x = int(float(outputFragmentCols) * outTex.x);
  int out_y = int(float(outputFragmentRows) * outTex.y);

  float sum = 0.;
  for (int n = 0; n < summationLength; ++n) {
    int rowIndex = texelFetch(rowIndexMap, ivec2(n, out_y), 0).r;
    int colIndex = texelFetch(colIndexMap, ivec2(n, out_y), 0).r;
    int fragmentIndex = int(floor(float(rowIndex) / float(inputFragmentRows)));
    if (rowIndex != -1 && colIndex != -1) {
      rowIndex = int(mod(float(rowIndex), float(inputFragmentRows)));
      colIndex += fragmentIndex * inputFragmentCols;
      sum += texelFetch(matMulResult, ivec2(colIndex + out_x, rowIndex), 0).r;
    }
  }

  if (use_bias) {
    sum += texelFetch(bias, ivec2(out_x, 0), 0).r;
    outColor = vec4(sum);
  } else {
    outColor = vec4(sum);
  }
}
