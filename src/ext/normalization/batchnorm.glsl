precision highp float;

varying vec2 outTex; // texture coords of row/column to calculate
uniform sampler2D X; // texture with data from padded X
uniform sampler2D mean;
uniform sampler2D std;
uniform sampler2D gamma;
uniform sampler2D beta;
uniform int N; // number of columns
uniform int pad; // additional columns to nearest multiple of four

// set pad values to 0.0, if in padded region of output texture
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

  // get the implied row and column from .y and .x of passed (output)
  // texture coordinate. These map directly to input texture space when
  // the relevant dimensions are the same.
  float row_t = outTex.y;
  float col_t = outTex.x;
  float col = (col_t * float(N + pad) - 2.0); // index of first element in pixel (matrix space)

  // direct usage of col requires output be padded exactly like input
  vec4 _x = texture2D(X, vec2(col_t, row_t));
  vec4 _mean = texture2D(mean, vec2(col_t, row_t));
  vec4 _std = texture2D(std, vec2(col_t, row_t));
  vec4 _gamma = texture2D(gamma, vec2(col_t, row_t));
  vec4 _beta = texture2D(beta, vec2(col_t, row_t));
  vec4 sum_v = _beta + _gamma * (_x - _mean) / sqrt(_std);

  // fix padded region
  if (pad > 0 && col + 4.0 > float(N)) {
    fix_pad(sum_v, pad);
  }

  gl_FragColor = sum_v;
}
