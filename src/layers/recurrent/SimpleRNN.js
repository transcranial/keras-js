import * as activations from '../../activations';
import Tensor from '../../Tensor';
import Layer from '../../Layer';
import { gemv } from 'ndarray-blas-level2';
import ops from 'ndarray-ops';
import cwise from 'cwise';

/**
 * SimpleRNN layer class
 */
export default class SimpleRNN extends Layer {
  /**
   * Creates a SimpleRNN layer
   * @param {number} attrs.outputDim - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.returnSequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.goBackwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'SimpleRNN';

    const {
      outputDim = 1,
      activation = 'tanh',
      returnSequences = false,
      goBackwards = false,
      stateful = false
    } = attrs;

    this.outputDim = outputDim;

    // keep this.activation for Bidirectional wrapper layer to use
    this.activation = activation;
    this.activationFunc = activations[activation];

    this.returnSequences = returnSequences;
    this.goBackwards = goBackwards;
    this.stateful = stateful;

    // Layer weights specification
    this.params = [ 'W', 'U', 'b' ];
  }

  _combine = cwise({
    args: [ 'array', 'array', 'array', 'array' ],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b;
    }
  });

  /**
   * Method for layer computational logic
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  call(x) {
    let currentX = new Tensor([], [ x.tensor.shape[1] ]);

    const dimHiddenState = this.weights['b'].tensor.shape[0];
    let currentHiddenState = this.stateful && this.currentHiddenState
      ? this.currentHiddenState
      : new Tensor([], [ dimHiddenState ]);
    let tempXH = new Tensor([], [ dimHiddenState ]);
    let tempHH = new Tensor([], [ dimHiddenState ]);
    let previousHiddenState = new Tensor([], [ dimHiddenState ]);

    this.hiddenStateSequence = new Tensor([], [ x.tensor.shape[0], dimHiddenState ]);

    const _clearTemp = () => {
      const tempTensors = [ tempXH, tempHH ];
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0));
    };

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor);

      gemv(1, this.weights['W'].tensor.transpose(1, 0), currentX.tensor, 1, tempXH.tensor);
      gemv(1, this.weights['U'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHH.tensor);
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b'].tensor);
      this.activationFunc(currentHiddenState);
    };

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i;
      ops.assign(currentX.tensor, x.tensor.pick(inputIndex, null));
      _clearTemp();
      _step();

      if (this.returnSequences) {
        ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor);
      }
    }

    if (this.returnSequences) {
      x.tensor = this.hiddenStateSequence.tensor;
    } else {
      x.tensor = currentHiddenState.tensor;
    }

    if (this.stateful) {
      this.currentHiddenState = currentHiddenState;
    }

    return x;
  }
}
