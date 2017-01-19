import Layer from '../../Layer';
import Tensor from '../../Tensor';
import ops from 'ndarray-ops';

/**
 * UpSampling3D layer class
 */
export default class UpSampling3D extends Layer {
  /**
   * Creates a UpSampling3D activation layer
   * @param {number} attrs.size - upsampling factor
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'UpSampling3D';

    const { size = [ 2, 2, 2 ], dimOrdering = 'tf' } = attrs;

    this.size = size;
    this.dimOrdering = dimOrdering;
  }

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    // convert to tf ordering
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0);
    }

    const inputShape = x.tensor.shape;
    const outputShape = [
      inputShape[0] * this.size[0],
      inputShape[1] * this.size[1],
      inputShape[2] * this.size[2],
      inputShape[3]
    ];
    let y = new Tensor([], outputShape);
    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        for (let k = 0; k < this.size[2]; k++) {
          ops.assign(y.tensor.lo(i, j, k, 0).step(this.size[0], this.size[1], this.size[2], 1), x.tensor);
        }
      }
    }
    x.tensor = y.tensor;

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(3, 0, 1, 2);
    }

    return x;
  }
}
