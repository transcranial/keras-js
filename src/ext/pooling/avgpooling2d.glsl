// 2D Average Pooling op.
// This is an extension of weblas.
// https://github.com/waylonflinn/weblas

precision highp float;

varying vec2 outTex;
uniform sampler2D X;
uniform sampler2D poolIndexMapping;
uniform int inputRows;
uniform int poolElements;
uniform int poolElementsPad;

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
  float poolIndexCoordX;
  vec4 poolIndices;
  int poolIndexRGBA;
  float poolIndex;
  vec4 mappedValues;
  float inputCoordY;
  vec4 currentSum = vec4(0.0, 0.0, 0.0, 0.0);
  int poolElementsEffective = poolElements;
  for (int i = 0; i < 100; i += 1) {
    if (i >= poolElements) {
      break;
    }

    poolIndexCoordX = (float(i) + 0.5) / float(poolElements + poolElementsPad);
    poolIndices = texture2D(poolIndexMapping, vec2(poolIndexCoordX, outTex.y));
    poolIndexRGBA = int(mod(float(i), 4.0));
    poolIndex = select_index(poolIndices, poolIndexRGBA);

    if (poolIndex != -1.0) {
      inputCoordY = (poolIndex + 0.5) / float(inputRows);
      mappedValues = texture2D(X, vec2(outTex.x, inputCoordY));
      currentSum = currentSum + mappedValues;
    } else {
      poolElementsEffective = poolElementsEffective - 1;
    }
  }

  currentSum = currentSum / float(poolElementsEffective);

  gl_FragColor = currentSum;
}
