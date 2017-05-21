// Merge op.
// Modes: 'sum', 'mul', 'ave', 'max'
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D inputs[8]; // array length must be fixed
uniform int numInputs;
uniform int modeCode;
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

  vec4 mergeValues = vec4(0.0, 0.0, 0.0, 0.0);
  if (modeCode == 1) {
    // mul
    mergeValues = vec4(1.0, 1.0, 1.0, 1.0);
  } else if (modeCode == 4) {
    // max
    const float min = -1.0e+08;
    mergeValues = vec4(min, min, min, min);
  }

  for (int i = 0; i < 8; i += 1) {
    if (i >= numInputs) {
      break;
    }

    if (modeCode == 0 || modeCode == 3) {
      // sum
      // ave
      mergeValues = mergeValues + texture2D(inputs[i], vec2(outTex.x, outTex.y));
    } else if (modeCode == 1) {
      // mul
      mergeValues = mergeValues * texture2D(inputs[i], vec2(outTex.x, outTex.y));
    } else if (modeCode == 4) {
      // max
      mergeValues = max(mergeValues, texture2D(inputs[i], vec2(outTex.x, outTex.y)));
    }
  }

  if (modeCode == 3) {
    // ave
    mergeValues = mergeValues / float(numInputs);
  }

  // set pad values to 0.0, if in padded region of output texture
  if (outputColPad > 0 && col + 4.0 > float(outputCols)) {
    fix_pad(mergeValues, outputColPad);
  }

  gl_FragColor = mergeValues;
}
