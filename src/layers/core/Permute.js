import Layer from '../../Layer'

/**
 * Permute layer class
 * Note there is no concept of batch size in these layers (single-batch), so dim numbers 1 less
 * i.e., dim 1 in keras corresponds to dim 0 here, etc.
 */
export default class Permute extends Layer {
  /**
   * Creates a Permute layer
   * @param {number[]} attrs.dims
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Permute'

    const { dims = [] } = attrs
    this.dims = dims.map(dim => dim - 1)
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    if (this.dims.length !== x.tensor.shape.length) {
      throw new Error(
        `${this.name} [Permute layer] The specified dims permutation must match the number of dimensions.`
      )
    }
    x.tensor = x.tensor.transpose(...this.dims)
    return x
  }
}
