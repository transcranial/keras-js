import ndarray from 'ndarray'
import ops from 'ndarray-ops'

/**
 * Function to throw error if specified shape is incompatible with data
 *
 * @param {number[]} data
 * @param {number[]} shape
 */

export function checkShape(data, shape) {
  if (data.length && shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
    throw new Error('Specified shape incompatible with data.')
  }
}

/**
 * Shuffle ndarray data layout for WebGL
 * - data for TEXTURE_2D_ARRAY or TEXTURE_3D laid out sequentially per-slice
 *
 * @param {TypedArray} typedarrayConstructor
 * @param {Object} arr - ndarray tensor
 * @param {number[]} shape
 */
export function data3DLayoutForGL(typedarrayConstructor, arr, shape) {
  // must shuffle data layout for webgl
  //
  const data = new typedarrayConstructor(arr.data.length)
  const slice = ndarray(new typedarrayConstructor(shape[0] * shape[1]), [shape[0], shape[1]])
  let offset = 0
  for (let i = 0; i < shape[2]; i++) {
    ops.assign(slice, arr.pick(null, null, i))
    data.set(slice.data, offset)
    offset += shape[0] * shape[1]
  }

  return data
}
