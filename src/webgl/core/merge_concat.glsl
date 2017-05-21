// Merge op.
// Modes: 'concat'
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D inputs[8]; // array length must be fixed
uniform int numInputs;
uniform int inputChannelStartIndices[8];
uniform int outputRows;
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
  float col = floor(outTex.x * float(outputCols + outputColPad) - 0.5);

  vec4 outValues = vec4(0.0, 0.0, 0.0, 0.0);
  int row = int(floor(outTex.y * float(outputRows)));
  float inputCoordY;
  for (int i = 0; i < 8; i += 1) {
    if (i >= numInputs) {
      break;
    }

    if (i == numInputs - 1) {
      inputCoordY = (0.5 + floor(outTex.y * float(outputRows)) - float(inputChannelStartIndices[i])) / float(outputRows - inputChannelStartIndices[i]);
      outValues = texture2D(inputs[i], vec2(outTex.x, inputCoordY));
      break;
    } else if (row >= inputChannelStartIndices[i] && row < inputChannelStartIndices[i + 1]) {
      inputCoordY = (0.5 + floor(outTex.y * float(outputRows)) - float(inputChannelStartIndices[i])) / float(inputChannelStartIndices[i + 1] - inputChannelStartIndices[i]);
      outValues = texture2D(inputs[i], vec2(outTex.x, inputCoordY));
      break;
    }
  }

  // set pad values to 0.0, if in padded region of output texture
  if (outputColPad > 0 && col + 4.0 > float(outputCols)) {
    fix_pad(outValues, outputColPad);
  }

  gl_FragColor = outValues;
}
