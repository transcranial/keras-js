import Layer from '../../Layer';
import Tensor from '../../Tensor';
import ops from 'ndarray-ops';

/**
 * Cropping2D layer class
 */
export default class Cropping2D extends Layer {
  /**
   * Creates a Cropping2D activation layer
   * @param {cropping} attrs.cropping - tuple of tuple of int (length 2)
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Cropping2D';

    const { cropping = [ [ 0, 0 ], [ 0, 0 ] ], dimOrdering = 'tf' } = attrs;

    this.cropping = cropping;
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
      x.tensor = x.tensor.transpose(1, 2, 0);
    }

    const inputShape = x.tensor.shape;
    const outputShape = [
      inputShape[0] - this.cropping[0][0] - this.cropping[0][1],
      inputShape[1] - this.cropping[1][0] - this.cropping[1][1],
      inputShape[2]
    ];
    let y = new Tensor([], outputShape);
    ops.assign(
      y.tensor,
      x.tensor
        .hi(inputShape[0] - this.cropping[0][1], inputShape[1] - this.cropping[1][1], inputShape[2])
        .lo(this.cropping[0][0], this.cropping[1][0], 0)
    );
    x.tensor = y.tensor;

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(2, 0, 1);
    }

    return x;
  }
}
