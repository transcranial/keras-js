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

void fix_pad(inout vec4 v, int pad) {
  v.a = 0.0;
  if (pad == 2) {
    v.b = 0.0;
  } else if (pad == 3) {
    v.b = 0.0;
    v.g = 0.0;
  }
}

void main(void) {
  // index of first element in pixel (matrix space)
  float col = floor(outTex.x * float(outputCols + outputColPad) - 1.5);

  vec4 _x = texture2D(X, vec2(outTex.x, outTex.y));
  vec4 _mean = texture2D(mean, vec2(outTex.x, 0.5));
  vec4 _std = texture2D(std, vec2(outTex.x, 0.5));
  vec4 _gamma = texture2D(gamma, vec2(outTex.x, 0.5));
  vec4 _beta = texture2D(beta, vec2(outTex.x, 0.5));
  vec4 sumValues = _beta + _gamma * (_x - _mean) / sqrt(_std + epsilon);

  // set pad values to 0.0, if in padded region of output texture
  if (outputColPad > 0 && col + 4.0 > float(outputCols)) {
    fix_pad(sumValues, outputColPad);
  }

  gl_FragColor = sumValues;
}
