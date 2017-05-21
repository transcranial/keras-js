// Transform input matrix X based on index mappings, indexMappingRow and indexMappingCol.
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform sampler2D indexMappingRow;
uniform sampler2D indexMappingCol;
uniform int inputRows;
uniform int inputCols;
uniform int outputCols;
uniform int inputColPad;
uniform int outputColPad;

float select_index(vec4 v, int index) {
  float val = 0.0;
  if (index == 0) {
    val = v.r;
  } else if (index == 1) {
    val = v.g;
  } else if (index == 2) {
    val = v.b;
  } else if (index == 3) {
    val = v.a;
  }
  return val;
}

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

  vec4 rowIndices = texture2D(indexMappingRow, vec2(outTex.x, outTex.y));
  vec4 colIndices = texture2D(indexMappingCol, vec2(outTex.x, outTex.y));

  float rowIndex;
  float colIndex;
  float inputCoordX;
  float inputCoordY;
  vec2 inputCoords;
  int inputChannel;
  vec4 mappedValues = vec4(0.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < 4; i++) {
    rowIndex = select_index(rowIndices, i);
    colIndex = select_index(colIndices, i);

    if (rowIndex != -1.0 && colIndex != -1.0) {
      inputCoordX = (float(colIndex) + 0.5) / float(inputCols + inputColPad);
      inputCoordY = (float(rowIndex) + 0.5) / float(inputRows);
      inputCoords = vec2(inputCoordX, inputCoordY);
      inputChannel = int(mod(colIndex, 4.0));
      if (i == 0) {
        mappedValues.r = select_index(texture2D(X, inputCoords), inputChannel);
      } else if (i == 1) {
        mappedValues.g = select_index(texture2D(X, inputCoords), inputChannel);
      } else if (i == 2) {
        mappedValues.b = select_index(texture2D(X, inputCoords), inputChannel);
      } else if (i == 3) {
        mappedValues.a = select_index(texture2D(X, inputCoords), inputChannel);
      }
    }

    // set pad values to 0.0, if in padded region of output texture
    if (outputColPad > 0 && col + 4.0 > float(outputCols)) {
      fix_pad(mappedValues, outputColPad);
    }
  }

  gl_FragColor = mappedValues;
}
