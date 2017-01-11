precision highp float;

varying vec2 outTex; // texture coords of row/column to calculate
uniform sampler2D X;
uniform sampler2D poolIndexMapping;
uniform int pad; // additional columns to nearest multiple of four
uniform int channels;
uniform int poolElements;

void main(void) {
  float row_t = outTex.y;
  float col = (outTex.x * float(channels + pad) - 2.0); // index of first element in pixel (matrix space)

  float poolIndexMapX;
  vec4 poolIndices;
  vec4 mapped_input_val;
  const float min = -1.0e+08;
  vec4 currentMax = vec4(min, min, min, min);
  for (int i = 0; i < 100; i += 1) {
    if (i >= poolElements) {
      break;
    }

    poolIndexMapX = float(i) / float(poolElements) + 0.5;
    poolIndices = texture2D(poolIndexMapping, vec2(poolIndexMapX, outTex.y));

    mapped_input_val = vec4(0.0, 0.0, 0.0, 0.0);
    mapped_input_val.r = texture2D(X, vec2(outTex.x, poolIndices.r)).r;
    if (pad > 0 && (col + 4.0) > float(channels)) {
      if (pad < 3) {
        mapped_input_val.g = texture2D(X, vec2(outTex.x, poolIndices.g)).g;
      }
      if (pad < 2) {
        mapped_input_val.b = texture2D(X, vec2(outTex.x, poolIndices.b)).b;
      }
    } else {
      mapped_input_val.g = texture2D(X, vec2(outTex.x, poolIndices.g)).g;
      mapped_input_val.b = texture2D(X, vec2(outTex.x, poolIndices.b)).b;
      mapped_input_val.a = texture2D(X, vec2(outTex.x, poolIndices.a)).a;
    }

    currentMax = max(currentMax, mapped_input_val);
  }

  gl_FragColor = currentMax;
}
