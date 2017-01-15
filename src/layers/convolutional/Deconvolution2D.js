import * as activations from '../../activations';
import Tensor from '../../Tensor';
import Layer from '../../Layer';
import ops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

/**
 * Deconvolution2D layer class
 */
export default class Deconvolution2D extends Layer {
  /**
   * Creates a Deconvolution2D layer
   * @param {number} attrs.nbFilter - Number of convolution filters to use.
   * @param {number} attrs.nbRow - Number of rows in the convolution kernel.
   * @param {number} attrs.nbCol - Number of columns in the convolution kernel.
   * @param {number[]} attrs.outputShape - Output shape of the transposed convolution operation.
   *   Array of integers [nbFilter, outputRows, outputCols]
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Deconvolution2D';

    const {
      nbFilter = 1,
      nbRow = 1,
      nbCol = 1,
      outputShape = [],
      activation = 'linear',
      borderMode = 'valid',
      subsample = [ 1, 1 ],
      dimOrdering = 'tf',
      bias = true
    } = attrs;

    this.kernelShape = [ nbFilter, nbRow, nbCol ];

    if (outputShape[0] == null) {
      this.outputShape = outputShape.slice(1);
    } else {
      this.outputShape = outputShape;
    }

    this.activation = activation;
    this.activationFunc = activations[activation];

    if (borderMode === 'valid' || borderMode === 'same') {
      this.borderMode = borderMode;
    } else {
      throw new Error(
        `${this.name} [Deconvolution2D layer] Invalid borderMode.`
      );
    }

    this.subsample = subsample;

    if (dimOrdering === 'tf' || dimOrdering === 'th') {
      this.dimOrdering = dimOrdering;
    } else {
      throw new Error(
        `${this.name} [Deconvolution2D layer] Only tf and th dim ordering are allowed.`
      );
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
   * In `tf` mode, W weight tensor has shape [nbRow, nbCol, inputChannels, nbFilter]
   * In `th` mode, W weight tensor has shape [nbFilter, inputChannels, nbRow, nbCol]
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dimOrdering === 'th') {
      // W
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0);
    }
    super.setWeights(weightsArr);

    this._wRowsMat = this._w2row();
    if (this._useWeblas) {
      this._wRowsMat.createWeblasTensor();
      if (!this._wRowsMat._gpuMaxSizeExceeded) {
        this._wRowsMat.weblasTensor = this._wRowsMat.weblasTensor.transpose();
      }
    }
  }

  /**
   * Method for computing output dimensions and padding, based on input
   * dimensions, kernel size, and padding mode.
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   * For deconvolution, we will "take away" padding from the output rather than add padding
   * to the input.
   * For more details on calculating output shapes and padding for transposed convolutions
   * (deconvolution here), see: https://arxiv.org/pdf/1603.07285v1.pdf
   * @param {Tensor} x
   */
  _calcOutputPadding(x) {
    const inputRows = x.tensor.shape[0];
    const inputCols = x.tensor.shape[1];
    const nbRow = this.kernelShape[1];
    const nbCol = this.kernelShape[2];

    // In contrast to Convolution2D, where we calculate the output shape,
    // the output shape is taken from the construtor variable, since
    // there is some level of ambiguity: input of shape [4, 4, inputChannels]
    // can have an output shape of either [7, 7, nbFilter] or [8, 8, nbFilter]
    // with borderMode `same` and subsample (stride) [2, 2].
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];

    const paddingRow = this.borderMode === 'same'
      ? Math.max(
        0,
        Math.floor((inputRows - 1) * this.subsample[0] + nbRow - outputRows)
      )
      : 0;
    const paddingCol = this.borderMode === 'same'
      ? Math.max(
        0,
        Math.floor((inputCols - 1) * this.subsample[1] + nbCol - outputCols)
      )
      : 0;
    const paddingRowBefore = Math.floor(paddingRow / 2);
    const paddingRowAfter = paddingRow - paddingRowBefore;
    const paddingColBefore = Math.floor(paddingCol / 2);
    const paddingColAfter = paddingCol - paddingColBefore;

    this.outputPadding = [
      paddingRowBefore,
      paddingRowAfter,
      paddingColBefore,
      paddingColAfter
    ];
  }

  /**
   * Convert input image to column matrix, along channels axis
   * shape: [inputRows, inputCols, inputChannels] -> [inputRows * inputCols, inputChannels]
   * @param {Tensor} x
   * @returns {Tensor} x
   */
  _im2col(x) {
    const [ inputRows, inputCols, inputChannels ] = x.tensor.shape;

    const imColsMat = new Tensor([], [ inputRows * inputCols, inputChannels ]);
    let channelRaveled = new Tensor([], [ inputRows * inputCols ]);
    let channel = new Tensor([], [ inputRows, inputCols ]);
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(channel.tensor, x.tensor.pick(null, null, c));
      channelRaveled.replaceTensorData(channel.tensor.data);
      ops.assign(imColsMat.tensor.pick(null, c), channelRaveled.tensor);
    }
    return imColsMat;
  }

  /**
   * Convert filter weights to row matrix, along channels axis
   * shape: [nbRow, nbCol, inputChannels, nbFilter] -> [inputChannels, nbRow * nbCol * nbFilter]
   * @returns {Tensor|weblas.pipeline.Tensor} wRowsMat
   */
  _w2row() {
    const [
      nbRow,
      nbCol,
      inputChannels,
      nbFilter
    ] = this.weights.W.tensor.shape;

    const wRowsMat = new Tensor([], [
      inputChannels,
      nbRow * nbCol * nbFilter
    ]);

    let channelRaveled = new Tensor([], [ nbRow * nbCol * nbFilter ]);
    let channel = new Tensor([], [ nbRow, nbCol, nbFilter ]);
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(
        channel.tensor,
        this.weights.W.tensor.pick(null, null, c, null)
      );
      channelRaveled.replaceTensorData(channel.tensor.data);
      ops.assign(wRowsMat.tensor.pick(c, null), channelRaveled.tensor);
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
      x.tensor = x.tensor.transpose(1, 2, 0);
    }

    const imColsMat = this._im2col(x);
    if (this._useWeblas) {
      imColsMat.createWeblasTensor();
    }

    const inputRows = x.tensor.shape[0];
    const inputCols = x.tensor.shape[1];
    const [ nbFilter, nbRow, nbCol ] = this.kernelShape;
    const matMul = new Tensor([], [
      inputRows * inputCols,
      nbRow * nbCol * nbFilter
    ]);

    if (
      this._useWeblas &&
        !(imColsMat._gpuMaxSizeExceeded || this._wRowsMat._gpuMaxSizeExceeded)
    ) {
      let _zerosVec = new Tensor([], [ this.weights.W.tensor.shape[3] ]);
      _zerosVec.createWeblasTensor();
      matMul.tensor.data = weblas.pipeline
        .sgemm(
          1,
          imColsMat.weblasTensor,
          this._wRowsMat.weblasTensor,
          0,
          _zerosVec
        )
        .transfer();
      imColsMat.weblasTensor.delete();
      delete imColsMat.weblasTensor;
    } else {
      gemm(matMul.tensor, imColsMat.tensor, this._wRowsMat.tensor, 1, 1);
    }

    this._calcOutputPadding(x);

    // add padding which we will take away later
    const [
      paddingRowBefore,
      paddingRowAfter,
      paddingColBefore,
      paddingColAfter
    ] = this.outputPadding;
    let output = new Tensor([], this.outputShape);
    let outputPadded = new Tensor([], [
      this.outputShape[0] + paddingRowBefore + paddingRowAfter,
      this.outputShape[1] + paddingColBefore + paddingColAfter,
      this.outputShape[2]
    ]);

    // bias
    if (this.bias) {
      for (let n = 0; n < nbFilter; n++) {
        ops.assigns(
          outputPadded.tensor.pick(null, null, n),
          this.weights.b.tensor.get(n)
        );
      }
    }

    const patchShape = [ nbRow, nbCol, nbFilter ];
    let patch = new Tensor([], patchShape);
    let patchRaveled = new Tensor([], [ nbRow * nbCol * nbFilter ]);
    let index = 0;
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        ops.assign(patchRaveled.tensor, matMul.tensor.pick(index, null));
        patch.replaceTensorData(patchRaveled.tensor.data);
        const iOutPos = i * this.subsample[0];
        const jOutPos = j * this.subsample[1];
        ops.addeq(
          outputPadded.tensor
            .hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2])
            .lo(iOutPos, jOutPos, 0),
          patch.tensor
        );
        index += 1;
      }
    }

    // remove padding
    ops.assign(
      output.tensor,
      outputPadded.tensor
        .hi(
          this.outputShape[0] + paddingRowBefore,
          this.outputShape[1] + paddingColBefore,
          this.outputShape[2]
        )
        .lo(paddingRowBefore, paddingColBefore, 0)
    );

    x.tensor = output.tensor;
    this.activationFunc(x);

    // convert back to th ordering if necessary
    if (this.dimOrdering === 'th') {
      x.tensor = x.tensor.transpose(2, 0, 1);
    }

    return x;
  }
}
