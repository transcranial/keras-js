import * as activations from '../../activations';
import Tensor from '../../Tensor';
import Layer from '../../Layer';
import ops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

/**
 * Convolution3D layer class
 */
export default class Convolution3D extends Layer {
  /**
   * Creates a Convolution3D layer
   * @param {number} attrs.nbFilter - Number of convolution filters to use.
   * @param {number} attrs.kernelDim1 - Length of the first dimension in the convolution kernel.
   * @param {number} attrs.kernelDim2 - Length of the second dimension in the convolution kernel.
   * @param {number} attrs.kernelDim3 - Length of the third dimension in the convolution kernel.
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Convolution3D';

    const {
      nbFilter = 1,
      kernelDim1 = 1,
      kernelDim2 = 1,
      kernelDim3 = 1,
      activation = 'linear',
      borderMode = 'valid',
      subsample = [ 1, 1, 1 ],
      dimOrdering = 'tf',
      bias = true
    } = attrs;

    this.kernelShape = [ nbFilter, kernelDim1, kernelDim2, kernelDim3 ];

    this.activation = activation;
    this.activationFunc = activations[activation];

    if (borderMode === 'valid' || borderMode === 'same') {
      this.borderMode = borderMode;
    } else {
      throw new Error(`${this.name} [Convolution3D layer] Invalid borderMode.`);
    }

    this.subsample = subsample;

    if (dimOrdering === 'tf' || dimOrdering === 'th') {
      this.dimOrdering = dimOrdering;
    } else {
      throw new Error(`${this.name} [Convolution3D layer] Only tf and th dim ordering are allowed.`);
    }

    this.bias = bias;

    // Layer weights specification
    this.params = this.bias ? [ 'W', 'b' ] : [ 'W' ];

    // Enable layer gpu +/- pipeline mode if supported
    if (this.gpu && weblas) {
      this._useWeblas = true;
      this._pipelineEnabled = false;
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * W weight tensor is converted to `tf` mode if in `th` mode.
   * In `tf` mode, W weight tensor has shape [kernelDim1, kernelDim2, kernelDim3, inputChannels, nbFilter]
   * In `th` mode, W weight tensor has shape [nbFilter, inputChannels, kernelDim1, kernelDim2, kernelDim3]
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dimOrdering === 'th') {
      // W
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 4, 1, 0);
    }
    super.setWeights(weightsArr);

    this._wRowsMat = this._w2row();
    if (this._useWeblas) {
      this._wRowsMat.createWeblasTensor();
      if (!this._wRowsMat._gpuMaxSizeExceeded) {
        this._wRowsMat.weblasTensor = this._wRowsMat.weblasTensor.transpose();
      }
      if (this.bias) {
        this.weights.b.createWeblasTensor();
      } else {
        this._zerosVec = new Tensor([], [ this.weights.W.tensor.shape[4] ]);
        this._zerosVec.createWeblasTensor();
      }
    }
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * @param {Tensor} x
   */
  _calcOutputShape(x) {
    const inputDim1 = x.tensor.shape[0];
    const inputDim2 = x.tensor.shape[1];
    const inputDim3 = x.tensor.shape[2];
    const [ nbFilter, kernelDim1, kernelDim2, kernelDim3 ] = this.kernelShape;

    const outputDim1 = this.borderMode === 'same'
      ? Math.floor((inputDim1 + this.subsample[0] - 1) / this.subsample[0])
      : Math.floor((inputDim1 - kernelDim1 + this.subsample[0]) / this.subsample[0]);
    const outputDim2 = this.borderMode === 'same'
      ? Math.floor((inputDim2 + this.subsample[1] - 1) / this.subsample[1])
      : Math.floor((inputDim2 - kernelDim2 + this.subsample[1]) / this.subsample[1]);
    const outputDim3 = this.borderMode === 'same'
      ? Math.floor((inputDim3 + this.subsample[2] - 1) / this.subsample[2])
      : Math.floor((inputDim3 - kernelDim3 + this.subsample[2]) / this.subsample[2]);
    const outputChannels = nbFilter;

    const paddingDim1 = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputDim1 - 1) * this.subsample[0] + kernelDim1 - inputDim1))
      : 0;
    const paddingDim2 = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputDim2 - 1) * this.subsample[1] + kernelDim2 - inputDim2))
      : 0;
    const paddingDim3 = this.borderMode === 'same'
      ? Math.max(0, Math.floor((outputDim3 - 1) * this.subsample[2] + kernelDim3 - inputDim3))
      : 0;
    const paddingDim1Before = Math.floor(paddingDim1 / 2);
    const paddingDim1After = paddingDim1 - paddingDim1Before;
    const paddingDim2Before = Math.floor(paddingDim2 / 2);
    const paddingDim2After = paddingDim2 - paddingDim2Before;
    const paddingDim3Before = Math.floor(paddingDim3 / 2);
    const paddingDim3After = paddingDim3 - paddingDim3Before;

    this.outputShape = [ outputDim1, outputDim2, outputDim3, outputChannels ];
    this.inputPadding = [
      paddingDim1Before,
      paddingDim1After,
      paddingDim2Before,
      paddingDim2After,
      paddingDim3Before,
      paddingDim3After
    ];
  }

  /**
   * Pad input tensor if necessary, for borderMode='same'
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _padInput(x) {
    if (this.borderMode === 'same') {
      const [ inputDim1, inputDim2, inputDim3, inputChannels ] = x.tensor.shape;
      const [
        paddingDim1Before,
        paddingDim1After,
        paddingDim2Before,
        paddingDim2After,
        paddingDim3Before,
        paddingDim3After
      ] = this.inputPadding;
      const newDim1 = inputDim1 + paddingDim1Before + paddingDim1After;
      const newDim2 = inputDim2 + paddingDim2Before + paddingDim2After;
      const newDim3 = inputDim3 + paddingDim3Before + paddingDim3After;
      let _x = new Tensor([], [ newDim1, newDim2, newDim3, inputChannels ]);
      ops.assign(
        _x.tensor
          .hi(
            inputDim1 + paddingDim1Before,
            inputDim2 + paddingDim2Before,
            inputDim3 + paddingDim3Before,
            inputChannels
          )
          .lo(paddingDim1Before, paddingDim2Before, paddingDim3Before, 0),
        x.tensor
      );
      x.tensor = _x.tensor;
    }
    return x;
  }

  /**
   * Convert input volume to column matrix
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _vol2col(x) {
    const [ inputDim1, inputDim2, inputDim3, inputChannels ] = x.tensor.shape;
    const kernelDim1 = this.kernelShape[1];
    const kernelDim2 = this.kernelShape[2];
    const kernelDim3 = this.kernelShape[3];
    const outputDim1 = this.outputShape[0];
    const outputDim2 = this.outputShape[1];
    const outputDim3 = this.outputShape[2];
    const nbPatches = outputDim1 * outputDim2 * outputDim3;
    const patchLen = kernelDim1 * kernelDim2 * kernelDim3 * inputChannels;

    if (!this._volColsMat) {
      this._volColsMat = new Tensor([], [ nbPatches, patchLen ]);
    }

    if (
      kernelDim1 === 1 &&
        kernelDim2 === 1 &&
        kernelDim3 === 1 &&
        this.subsample[0] === 1 &&
        this.subsample[1] === 1 &&
        this.subsample[2] === 1
    ) {
      this._volColsMat.replaceTensorData(x.tensor.data);
      if (this._useWeblas) {
        this._volColsMat.createWeblasTensor();
      }
      return this._volColsMat;
    }

    let patch = new Tensor([], [ kernelDim1, kernelDim2, kernelDim3, inputChannels ]);
    let offset = 0;
    for (let i = 0, limit = inputDim1 - kernelDim1; i <= limit; i += this.subsample[0]) {
      for (let j = 0, limit = inputDim2 - kernelDim2; j <= limit; j += this.subsample[1]) {
        for (let k = 0, limit = inputDim3 - kernelDim3; k <= limit; k += this.subsample[2]) {
          ops.assign(
            patch.tensor,
            x.tensor.hi(i + kernelDim1, j + kernelDim2, k + kernelDim3, inputChannels).lo(i, j, k, 0)
          );
          this._volColsMat.tensor.data.set(patch.tensor.data, offset);
          offset += patchLen;
        }
      }
    }
    if (this._useWeblas) {
      this._volColsMat.createWeblasTensor();
    }
    return this._volColsMat;
  }

  /**
   * Convert filter weights to row matrix
   * @returns {Tensor|weblas.pipeline.Tensor} wRowsMat
   */
  _w2row() {
    const inputChannels = this.weights.W.tensor.shape[3];
    const [ nbFilter, kernelDim1, kernelDim2, kernelDim3 ] = this.kernelShape;
    const patchLen = kernelDim1 * kernelDim2 * kernelDim3 * inputChannels;

    const wRowsMat = new Tensor([], [ patchLen, nbFilter ]);

    let patch = new Tensor([], [ kernelDim1, kernelDim2, kernelDim3, inputChannels ]);
    let patchRaveled = new Tensor([], [ patchLen ]);
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(patch.tensor, this.weights.W.tensor.pick(null, null, null, null, n));
      patchRaveled.replaceTensorData(patch.tensor.data);
      ops.assign(wRowsMat.tensor.pick(null, n), patchRaveled.tensor);
    }

    return wRowsMat;
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

    this._calcOutputShape(x);
    this._padInput(x);

    this._vol2col(x);

    const nbFilter = this.kernelShape[0];
    const outputDim1 = this.outputShape[0];
    const outputDim2 = this.outputShape[1];
    const outputDim3 = this.outputShape[2];
    const nbPatches = outputDim1 * outputDim2 * outputDim3;
    const matMul = new Tensor([], [ nbPatches, nbFilter ]);

    if (this._useWeblas && !(this._volColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)) {
      const bias = this.bias ? this.weights.b.weblasTensor : this._zerosVec.weblasTensor;
      matMul.tensor.data = weblas.pipeline
        .sgemm(1, this._volColsMat.weblasTensor, this._wRowsMat.weblasTensor, 1, bias)
        .transfer();
    } else {
      if (this.bias) {
        for (let n = 0; n < nbFilter; n++) {
          ops.assigns(matMul.tensor.pick(null, n), this.weights.b.tensor.get(n));
        }
      }
      gemm(matMul.tensor, this._volColsMat.tensor, this._wRowsMat.tensor, 1, 1);
    }

    let output = new Tensor([], this.outputShape);
    let outputChannelRaveled = new Tensor([], [ outputDim1 * outputDim2 * outputDim3 ]);
    let outputChannel = new Tensor([], [ outputDim1, outputDim2, outputDim3 ]);
    for (let n = 0; n < nbFilter; n++) {
      ops.assign(outputChannelRaveled.tensor, matMul.tensor.pick(null, n));
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data);
      ops.assign(output.tensor.pick(null, null, null, n), outputChannel.tensor);
    }
    x.tensor = output.tensor;

    this.activationFunc(x);

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(3, 0, 1, 2);
    }

    return x;
  }
}
