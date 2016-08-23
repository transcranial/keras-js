import * as activations from '../activations'
import Tensor from '../tensor'
import { Layer } from '../engine/topology'
import ndarray from 'ndarray'
import { gemv } from 'ndarray-blas-level2'
import gemm from 'ndarray-gemm'
import ops from 'ndarray-ops'
import unpack from 'ndarray-unpack'
import unsqueeze from 'ndarray-unsqueeze'
import tile from 'ndarray-tile'
import concatFirstAxis from 'ndarray-concat-rows'
import flattenDeep from 'lodash/flattenDeep'
import isEqual from 'lodash/isEqual'
import isInteger from 'lodash/isInteger'

/**
* Dense layer class
*/
export class Dense extends Layer {
  /**
  * Creates a Dense layer
  * @param {number} outputDim - output dimension size
  * @param {Object} [attrs] - layer attributes
  */
  constructor (outputDim, attrs = {}) {
    super(attrs)
    const {
      activation = 'linear',
      inputDim = null,
      bias = true
    } = attrs

    this.activation = activations[activation]
    this.outputDim = outputDim
    this.inputDim = inputDim
    this.bias = bias

    /**
    * Layer weights specification
    */
    this.params = this.bias ? ['W', 'b'] : ['W']

    /**
    * Input shape specification
    */
    if (this.inputDim) {
      this.inputShape = [this.inputDim]
    }
  }

  /**
  * Method for layer computational logic
  *
  * x = W^T * x + b
  *
  * weblas notes:
  * sgemm(M, N, K, alpha, A, B, beta, C), where A, B, C are Float32Array
  * - alpha * A * B + beta * C
  * - A has shape M x N
  * - B has shape N x K
  * - C has shape M x K
  * pipeline.sgemm(alpha, A, B, beta, C), where A, B, C are weblas.pipeline.Tensor here
  * - alpha * A * B^T + beta * C
  *
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    if (x._useWeblas) {
      // x is mutable, so create on every call
      x.createWeblasTensor()
      if (!this.weblasWeights) {
        // layer weights are immutable, so only create if not already existing
        this.createWeblasWeights()
      }

      const bias = this.bias
        ? this.weblasWeights.b
        : new weblas.pipeline.Tensor([1, this.outputDim], new Float32Array(this.outputDim))

      x.weblasTensor = weblas.pipeline.sgemm(
        1.0,
        x.weblasTensor,
        this.weblasWeights.W.transpose(true),
        1.0,
        bias
      )

      // activation function in CPU memory
      x.transferWeblasTensor()
    } else {
      let y = new Tensor([], [this.outputDim])
      if (this.bias) {
        ops.assign(y.tensor, this.weights.b.tensor)
      }
      gemv(1.0, this.weights.W.tensor.transpose(1, 0), x.tensor, 1.0, y.tensor)
      x.tensor = y.tensor
    }

    this.activation(x)

    return x
  }
}

/**
* Activation layer class
*/
export class Activation extends Layer {
  /**
  * Creates an Activation layer
  * @param {string} activation - name of activation function
  */
  constructor (activation, attrs = {}) {
    super({})
    this.activation = activations[activation]
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this.activation(x)
    return x
  }
}

/**
* Dropout layer class
* Note that this layer is here for compatibility, it's only applied during training time.
*/
export class Dropout extends Layer {
  /**
  * Creates an Dropout layer
  * @param {number} p - fraction of the input units to drop (between 0 and 1)
  */
  constructor (p) {
    super({})
    this.p = Math.min(Math.max(0, p), 1)
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    return x
  }
}

/**
* Flatten layer class
* Turns tensor into 1-d. Note there is no concept of batch size in these layers (single-batch).
* We use ndarray-unpack first, as ndarray striding/offsets precludes us from simply using x.tensor.data
*/
export class Flatten extends Layer {
  /**
  * Creates a Flatten layer
  */
  constructor () {
    super({})
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    if (x.tensor.shape.length > 1) {
      const shape = [x.tensor.shape.reduce((a, b) => a * b, 1)]
      x.tensor = ndarray(new x._type(flattenDeep(unpack(x.tensor))), shape)
    }
    return x
  }
}

/**
* Reshape layer class
* Note there is no concept of batch size in these layers (single-batch).
* We use ndarray-unpack first, as ndarray striding/offsets precludes us from simply using x.tensor.data
*/
export class Reshape extends Layer {
  /**
  * Creates a Reshape layer
  * @param {number[]} shape
  */
  constructor (shape) {
    super({})
    this.shape = shape
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    if (this.shape.reduce((a, b) => a * b, 1) !== x.tensor.size) {
      throw new Error(`${this.name} [Reshape layer] The total size of new array must be unchanged in reshape layer.`)
    }
    x.tensor = ndarray(new x._type(flattenDeep(unpack(x.tensor))), this.shape)
    return x
  }
}

/**
* Permute layer class
* Note there is no concept of batch size in these layers (single-batch), so dim numbers 1 less
* i.e., dim 1 in keras corresponds to dim 0 here, etc.
*/
export class Permute extends Layer {
  /**
  * Creates a Permute layer
  * @param {number[]} dims
  */
  constructor (dims) {
    super({})
    this.dims = dims.map(dim => dim - 1)
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    if (this.dims.length !== x.tensor.shape.length) {
      throw new Error(`${this.name} [Permute layer] The specified dims permutation must match the number of dimensions.`)
    }
    x.tensor = x.tensor.transpose(...this.dims)
    return x
  }
}

/**
* RepeatVector layer class
* Turns 2D tensors of shape [features] to 3D tensors of shape [n, features].
* Note there is no concept of batch size in these layers (single-batch) so we're actually going from 1D to 2D.
*/
export class RepeatVector extends Layer {
  /**
  * Creates a RepeatVector layer
  * @param {number} n
  */
  constructor (n) {
    super({})
    this.n = n
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    if (x.tensor.shape.length !== 1) {
      throw new Error(`${this.name} [RepeatVector layer] Only 1D tensor inputs allowed.`)
    }
    x.tensor = tile(unsqueeze(x.tensor, 0), [this.n, 1])
    return x
  }
}

/**
* Merge layer class
*/
export class Merge extends Layer {
  /**
  * Creates a Merge layer
  * @param {Object} [attrs] - layer attributes
  */
  constructor (attrs = {}) {
    super(attrs)
    const {
      mode = 'sum',
      concatAxis = -1,
      dotAxes = -1,
      outputShape = null,
      outputMask = null
    } = attrs

    const availableModes = ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max']
    if (availableModes.indexOf(mode) > -1) {
      this.mode = mode
    } else {
      throw new Error(`${this.name} [Merge layer] ${mode} not available.`)
    }

    this.concatAxis = concatAxis
    this.dotAxes = dotAxes
    this.outputShape = outputShape
    this.outputMask = outputMask
  }

  /**
  * Internal method for validating inputs
  * @param {Tensor[]} inputs
  * @returns {boolean} valid
  */
  _validateInputs = inputs => {
    const shapes = inputs.map(x => x.tensor.shape.slice())
    if (['sum', 'mul', 'ave', 'cos', 'max'].indexOf(this.mode) > -1) {
      if (!shapes.every(shape => isEqual(shape, shapes[0]))) {
        throw new Error(`${this.name} [Merge layer] All input shapes must be the same for mode ${this.mode}.`)
      }
    }
    if (['cos', 'dot'].indexOf(this.mode) > -1) {
      if (inputs.length !== 2) {
        throw new Error(`${this.name} [Merge layer] Exactly 2 inputs required for mode ${this.mode}.`)
      }
      if (isInteger(this.dotAxes)) {
        if (this.dotAxes < 0) {
          this.dotAxes = [shapes[0].length + this.dotAxes, shapes[1].length + this.dotAxes]
        } else {
          this.dotAxes = [this.dotAxes - 1, this.dotAxes - 1]
        }
      }
      if (shapes[0][this.dotAxes[0]] !== shapes[1][this.dotAxes[1]]) {
        throw new Error(`${this.name} [Merge layer] Dimensions incompatibility using dot mode.`)
      }
    } else if (this.mode === 'concat') {
      const nonConcatShapes = shapes.slice()
      nonConcatShapes.forEach(shape => shape.splice(this.concatAxis, 1))
      if (!nonConcatShapes.every(shape => isEqual(shape, nonConcatShapes[0]))) {
        throw new Error(`${this.name} [Merge layer] In concat mode, all shapes must be the same except along the concat axis.`)
      }
    }
    return true
  }

  /**
  * Method for layer computational logic
  * @param {Tensor[]} inputs
  * @returns {Tensor} `this`
  */
  call = inputs => {
    const valid = this._validateInputs(inputs)
    if (!valid) {
      throw new Error(`${this.name} [Merge layer] Invalid inputs to call method.`)
    }

    let output
    let outputShape
    if (['sum', 'mul', 'ave', 'max'].indexOf(this.mode) > -1) {
      outputShape = inputs[0].tensor.shape.slice()
      output = new Tensor([], outputShape)
    } else if (this.mode === 'concat') {
      outputShape = inputs[0].tensor.shape.slice()
      const _concatAxis = this.concatAxis < 0
        ? outputShape.length + this.concatAxis
        : this.concatAxis - 1
      inputs.slice(1, inputs.length).forEach(x => {
        const d = x.tensor.shape.slice()[_concatAxis]
        outputShape[_concatAxis] += d
      })
      output = new Tensor([], outputShape)
    } else if (['cos', 'dot'].indexOf(this.mode) > -1) {
      let shape1 = inputs[0].tensor.shape.slice()
      let shape2 = inputs[1].tensor.shape.slice()
      shape1.splice(this.dotAxes[0], 1)
      shape2.splice(this.dotAxes[1], 1)
      outputShape = shape1.concat(shape2)
      if (outputShape.length === 1) {
        outputShape.push(1)
      }
      output = new Tensor([], outputShape)
    }

    if (this.mode === 'sum') {
      for (let i = 0; i < inputs.length; i++) {
        ops.addeq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'mul') {
      ops.assigns(output.tensor, 1.0)
      for (let i = 0; i < inputs.length; i++) {
        ops.muleq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'ave') {
      for (let i = 0; i < inputs.length; i++) {
        ops.addeq(output.tensor, inputs[i].tensor)
      }
      ops.divseq(output.tensor, inputs.length)
    } else if (this.mode === 'max') {
      ops.assign(output.tensor, inputs[0].tensor)
      for (let i = 1; i < inputs.length; i++) {
        ops.maxeq(output.tensor, inputs[i].tensor)
      }
    } else if (this.mode === 'concat') {
      const _concatAxis = this.concatAxis < 0
        ? inputs[0].tensor.shape.length + this.concatAxis
        : this.concatAxis - 1
      if (_concatAxis === 0) {
        concatFirstAxis(output.tensor, inputs.map(x => x.tensor))
      } else {
        let dimsAxisSwap = [_concatAxis]
        for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
          if (i !== _concatAxis) dimsAxisSwap.push(i)
        }
        concatFirstAxis(
          output.tensor.transpose(...dimsAxisSwap),
          inputs.map(x => x.tensor.transpose(...dimsAxisSwap))
        )
      }
    } else if (this.mode === 'dot') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          gemm(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          gemm(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
        }
      } else {
        throw new Error(`${this.name} [Merge layer] dot mode for 3+ dim tensors not yet implemented.`)
      }
    } else if (this.mode === 'cos') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        let a = new Tensor([], output.tensor.shape)
        let b = new Tensor([], output.tensor.shape)
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          gemm(a.tensor, inputs[0].tensor.transpose(1, 0), inputs[0].tensor)
          gemm(b.tensor, inputs[1].tensor.transpose(1, 0), inputs[1].tensor)
          gemm(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor)
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          gemm(a.tensor, inputs[0].tensor, inputs[0].tensor.transpose(1, 0))
          gemm(b.tensor, inputs[1].tensor, inputs[1].tensor.transpose(1, 0))
          gemm(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0))
        }
        ops.muleq(a.tensor, b.tensor)
        ops.sqrteq(a.tensor)
        ops.diveq(output.tensor, a.tensor)
        output.tensor = unsqueeze(output.tensor, 0)
      } else {
        throw new Error(`${this.name} [Merge layer] cos mode for 3+ dim tensors not yet implemented.`)
      }
    }

    return output
  }
}
