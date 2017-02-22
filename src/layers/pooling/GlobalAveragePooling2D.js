import Layer from '../../Layer';
import Tensor from '../../Tensor';
import ops from 'ndarray-ops';

/**
 * GlobalAveragePooling2D layer class
 */
export default class GlobalAveragePooling2D extends Layer {
  /**
   * Creates a GlobalAveragePooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalAveragePooling2D';

    const { dimOrdering = 'tf' } = attrs;
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

    const [rows, cols, channels] = x.tensor.shape;
    let y = new Tensor([], [channels]);
    for (let i = 0, len = channels; i < len; i++) {
      y.tensor.set(i, ops.sum(x.tensor.pick(null, null, i)) / (rows * cols));
    }
    x.tensor = y.tensor;
    return x;
  }
}
