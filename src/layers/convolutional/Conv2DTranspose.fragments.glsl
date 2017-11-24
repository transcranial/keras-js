#version 300 es
precision highp float;
precision highp isampler2D;

in vec2 outTex;
uniform sampler2D matMulResult;
uniform isampler2D indexMap;
uniform sampler2D bias;
uniform bool use_bias;
uniform int inputFragmentCols;
uniform int outputFragmentCols;
out vec4 outColor;

void main() {
  int inputFragmentRows = textureSize(matMulResult, 0)[1];
  int outputFragmentRows = textureSize(indexMap, 0)[1];
  int summationLength = textureSize(indexMap, 0)[0];
  int out_x = int(float(outputFragmentCols) * outTex.x);
  int out_y = int(float(outputFragmentRows) * outTex.y);

  float sum = 0.;
  for (int n = 0; n < summationLength; ++n) {
    int index = texelFetch(indexMap, ivec2(n, out_y), 0).r;
    if (index != -1) {
      int rowIndex = int(floor(float(index) / float(inputFragmentCols)));
      int colIndex = int(mod(float(index), float(inputFragmentCols)));
      int fragmentIndex = int(floor(float(rowIndex) / float(inputFragmentRows)));
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
