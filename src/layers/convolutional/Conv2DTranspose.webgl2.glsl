#version 300 es
precision mediump float;
precision mediump int;
precision mediump isampler2DArray;

in vec2 outTex;
uniform sampler2D matMulOutput;
uniform isampler2DArray rowIndicesMap;
uniform isampler2DArray colIndicesMap;
uniform sampler2D bias;
uniform bool use_bias;
uniform int rows;
uniform int cols;
uniform int summationLength;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);

  float sum = 0.;
  for (int n = 0; n < summationLength; ++n) {
    int i = texelFetch(rowIndicesMap, ivec3(out_x, out_y, n), 0).r;
    int j = texelFetch(colIndicesMap, ivec3(out_x, out_y, n), 0).r;
    if (i >= 0 && j >= 0) {
      sum += texelFetch(matMulOutput, ivec2(j, i), 0).r;
    }
  }

  if (use_bias) {
    sum += texelFetch(bias, ivec2(out_x, 0), 0).r;
    outColor = vec4(sum);
  } else {
    outColor = vec4(sum);
  }
}
