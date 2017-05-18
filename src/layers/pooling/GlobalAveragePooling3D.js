import Layer from '../../Layer'
import Tensor from '../../Tensor'
import ops from 'ndarray-ops'

/**
 * GlobalAveragePooling3D layer class
 */
export default class GlobalAveragePooling3D extends Layer {
  /**
   * Creates a GlobalAveragePooling3D layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GlobalAveragePooling3D'

    const { data_format = 'channels_last' } = attrs
    this.dataFormat = data_format
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0)
    }

    const [dim1, dim2, dim3, channels] = x.tensor.shape
    let y = new Tensor([], [channels])
    for (let i = 0, len = channels; i < len; i++) {
      y.tensor.set(i, ops.sum(x.tensor.pick(null, null, null, i)) / (dim1 * dim2 * dim3))
    }
    x.tensor = y.tensor
    return x
  }
}
