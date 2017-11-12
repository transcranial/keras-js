#version 300 es
precision highp float;

in vec2 outTex;
uniform sampler2D featureMaps;
uniform sampler2D weights;
uniform int rows;
uniform int cols;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);
  int channels = textureSize(weights, 0)[0];

  int featureMapsRow = out_x + cols * out_y;

  float f_sum = 0.0;
  float w_sum = 0.0;
  for (int c = 0; c < channels; ++c) {
    float f = texelFetch(featureMaps, ivec2(c, featureMapsRow), 0).r;
    float w = texelFetch(weights, ivec2(c, 0), 0).r;
    f_sum += f * w;
    w_sum += w;
  }

  outColor = vec4(max(f_sum / w_sum, 0.0));
}
