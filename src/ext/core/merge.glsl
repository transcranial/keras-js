// Merge op.
// Modes: 'sum', 'mul', 'ave', 'max'
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D inputs[8]; // array length must be fixed
uniform int numInputs;
uniform int modeCode;

void main(void) {
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

  gl_FragColor = mergeValues;
}
