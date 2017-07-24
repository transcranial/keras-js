#version 300 es
precision mediump float;

in vec2 outTex;
uniform sampler2D X;
uniform sampler2D gamma;
uniform sampler2D beta;
uniform sampler2D mean;
uniform sampler2D std;
uniform float epsilon;
uniform int rows;
uniform int cols;
uniform bool scale;
uniform bool center;
out vec4 outColor;

void main() {
  int out_x = int(float(cols) * outTex.x);
  int out_y = int(float(rows) * outTex.y);

  float _x = texelFetch(X, ivec2(out_x, out_y), 0).r;
  float _mean = texelFetch(mean, ivec2(out_x, 0), 0).r;
  float _std = texelFetch(std, ivec2(out_x, 0), 0).r;

  float _gamma = 1.0;
  if (scale) {
    _gamma = texelFetch(gamma, ivec2(out_x, 0), 0).r;
  }

  float _beta = 0.0;
  if (center) {
    _beta = texelFetch(beta, ivec2(out_x, 0), 0).r;
  }

  float sum = _beta + _gamma * (_x - _mean) / sqrt(_std + epsilon);

  outColor = vec4(sum);
}
