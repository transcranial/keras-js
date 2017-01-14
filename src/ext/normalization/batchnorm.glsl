// Batch normalization op.
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform sampler2D gamma;
uniform sampler2D beta;
uniform sampler2D mean;
uniform sampler2D std;
uniform float epsilon;
uniform int outputCols;
uniform int outputColPad;

void main(void) {
  vec4 _x = texture2D(X, vec2(outTex.x, outTex.y));
  vec4 _mean = texture2D(mean, vec2(outTex.x, 0.5));
  vec4 _std = texture2D(std, vec2(outTex.x, 0.5));
  vec4 _gamma = texture2D(gamma, vec2(outTex.x, 0.5));
  vec4 _beta = texture2D(beta, vec2(outTex.x, 0.5));
  vec4 sumValues = _beta + _gamma * (_x - _mean) / sqrt(_std + epsilon);

  gl_FragColor = sumValues;
}
