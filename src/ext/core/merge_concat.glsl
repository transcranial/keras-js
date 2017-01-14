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

void main(void) {
  vec4 outValues = vec4(0.0, 0.0, 0.0, 0.0);
  int row = int(floor(outTex.y * float(outputRows)));
  float inputCoordY;
  for (int i = 0; i < 8; i += 1) {
    if (i >= numInputs) {
      break;
    }

    if (i == numInputs - 1) {
      inputCoordY = (floor(outTex.y * float(outputRows)) - float(inputChannelStartIndices[i])) / float(outputRows - inputChannelStartIndices[i]);
      outValues = texture2D(inputs[i], vec2(outTex.x, inputCoordY));
      break;
    } else if (row >= inputChannelStartIndices[i] && row < inputChannelStartIndices[i + 1]) {
      inputCoordY = (floor(outTex.y * float(outputRows)) - float(inputChannelStartIndices[i])) / float(inputChannelStartIndices[i + 1] - inputChannelStartIndices[i]);
      outValues = texture2D(inputs[i], vec2(outTex.x, inputCoordY));
      break;
    }
  }

  gl_FragColor = outValues;
}
