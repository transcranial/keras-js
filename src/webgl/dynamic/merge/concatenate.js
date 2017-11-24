import _ from 'lodash'

/**
 * Create GLSL program for merge.Concatenate layer
 *
 * @param {number} numInputs
 * @param {number[]} inputShape
 * @param {number[]} outputShape
 */
export default function concatenate(numInputs, inputShape, outputShape, concatAxis) {
  let block = 'outColor = vec4(0.0);'

  if (concatAxis === 0) {
    // prettier-ignore
    block = `
  int n = int(floor(float(out_y) / float(${inputShape[0]})));
  if (n == 0) {
    outColor = vec4(texelFetch(inputs[0], ivec2(out_x, out_y), 0).r);
  }${_.range(1, numInputs).map(
    i => ` else if (n == ${i}) {
    outColor = vec4(texelFetch(inputs[${i}], ivec2(out_x, out_y - ${i * inputShape[0]}), 0).r);
  }`).join('')}
`
  } else if (concatAxis === 1) {
    // prettier-ignore
    block = `
  int n = int(floor(float(out_x) / float(${inputShape[1]})));
  if (n == 0) {
    outColor = vec4(texelFetch(inputs[0], ivec2(out_x, out_y), 0).r);
  }${_.range(1, numInputs).map(
    i => ` else if (n == ${i}) {
    outColor = vec4(texelFetch(inputs[${i}], ivec2(out_x - ${i * inputShape[1]}, out_y), 0).r);
  }`).join('')}
`
  }

  const source = `#version 300 es
precision highp float;

in vec2 outTex;
uniform sampler2D inputs[${numInputs}];
out vec4 outColor;

void main() {
  int out_y = int(float(${outputShape[0]}) * outTex.y);
  int out_x = int(float(${outputShape[1]}) * outTex.x);
${block}
}
`

  return source
}
