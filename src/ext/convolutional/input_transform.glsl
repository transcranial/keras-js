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
uniform int inputColPad;

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

void main(void) {
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
  }

  gl_FragColor = mappedValues;
}
