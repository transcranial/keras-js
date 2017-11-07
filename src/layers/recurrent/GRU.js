import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { webgl2 } from '../../WebGL2'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'
import cwise from 'cwise'

/**
 * GRU layer class
 */
export default class GRU extends Layer {
  /**
   * Creates a GRU layer
   *
   * @param {Object} [attrs] - layer attributes
   * @param {number} [attrs.units] - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.recurrent_activation] - inner activation function
   * @param {number} [attrs.use_bias] - use bias
   * @param {number} [attrs.return_sequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.go_backwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'GRU'

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      recurrent_activation = 'hard_sigmoid',
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs

    this.units = units

    // keep this.activation and this.recurrentActivation for Bidirectional wrapper layer to use
    this.activation = activation
    this.recurrentActivation = recurrent_activation
    this.activationFunc = activations[activation]
    this.recurrentActivationFunc = activations[recurrent_activation]

    this.use_bias = use_bias

    this.returnSequences = return_sequences
    this.goBackwards = go_backwards
    this.stateful = stateful

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'recurrent_kernel', 'bias'] : ['kernel', 'recurrent_kernel']

    // GPU setup
    if (this.gpu) {
      this.copyTextureProgram = webgl2.compileProgram(require('../../copyTexture.glsl'))
      this.matMulProgram = webgl2.compileProgram(require('../../matMul.glsl'))
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`))
      this.recurrentActivationProgram = webgl2.compileProgram(
        require(`../../activations/${this.recurrentActivation}.glsl`)
      )
      this.gateSummationProgram = webgl2.compileProgram(require('./gateSummation.glsl'))
      this.gateProductProgram = webgl2.compileProgram(require('./gateProduct.glsl'))
      this.timestepReadProgram = webgl2.compileProgram(require('./timestepRead.glsl'))
      this.timestepWriteProgram = webgl2.compileProgram(require('./timestepWrite.glsl'))
      this.updateProgram = webgl2.compileProgram(require('./GRU.update.glsl'))
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * W weight tensor is split into W_z, W_r, W_h
   *
   * U weight tensor is split into U_z, U_r, U_h
   *
   * b weight tensor is split into b_z, b_r, b_h (or create empty bias if this.use_bias is false)
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    const shape_W = this.weights['kernel'].tensor.shape
    this.weights['W_z'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_r'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_h'] = new Tensor([], [shape_W[0], this.units])
    ops.assign(this.weights['W_z'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['W_r'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['W_h'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 3 * this.units).lo(0, 2 * this.units)
    )

    const shape_U = this.weights['recurrent_kernel'].tensor.shape
    this.weights['U_z'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_r'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_h'] = new Tensor([], [shape_U[0], this.units])
    ops.assign(this.weights['U_z'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['U_r'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['U_h'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 3 * this.units).lo(0, 2 * this.units)
    )

    this.weights['b_z'] = new Tensor([], [this.units])
    this.weights['b_r'] = new Tensor([], [this.units])
    this.weights['b_h'] = new Tensor([], [this.units])
    if (this.use_bias) {
      ops.assign(this.weights['b_z'].tensor, this.weights['bias'].tensor.hi(this.units).lo(0))
      ops.assign(this.weights['b_r'].tensor, this.weights['bias'].tensor.hi(2 * this.units).lo(this.units))
      ops.assign(this.weights['b_h'].tensor, this.weights['bias'].tensor.hi(3 * this.units).lo(2 * this.units))
    }

    if (this.gpu) {
      const names = ['W_z', 'W_r', 'W_h', 'U_z', 'U_r', 'U_h', 'b_z', 'b_r', 'b_h']
      names.forEach(name => {
        this.weights[name].createGLTexture()
      })
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  _update = cwise({
    args: ['array', 'array', 'array'],
    body: function(_h, _htm1, _z) {
      _h = _h * (1 - _z) + _htm1 * _z
    }
  })

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    const dimUpdateGate = this.weights['b_z'].tensor.shape[0]
    const dimResetGate = this.weights['b_r'].tensor.shape[0]
    const dimHiddenState = this.weights['b_h'].tensor.shape[0]

    const currentUpdateGateState = new Tensor([], [dimUpdateGate])
    const tempXZ = new Tensor([], [dimUpdateGate])
    const tempHZ = new Tensor([], [dimUpdateGate])

    const currentResetGateState = new Tensor([], [dimResetGate])
    const tempXR = new Tensor([], [dimResetGate])
    const tempHR = new Tensor([], [dimResetGate])

    const currentHiddenState =
      this.stateful && this.currentHiddenState ? this.currentHiddenState : new Tensor([], [dimHiddenState])
    const tempXH = new Tensor([], [dimHiddenState])
    const tempHH = new Tensor([], [dimHiddenState])
    const previousHiddenState = new Tensor([], [dimHiddenState])

    this.hiddenStateSequence = new Tensor([], [x.tensor.shape[0], dimHiddenState])

    const currentX = new Tensor([], [x.tensor.shape[1]])

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1, this.weights['W_z'].tensor.transpose(1, 0), currentX.tensor, 1, tempXZ.tensor)
      gemv(1, this.weights['U_z'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHZ.tensor)
      this._combine(currentUpdateGateState.tensor, tempXZ.tensor, tempHZ.tensor, this.weights['b_z'].tensor)
      this.recurrentActivationFunc(currentUpdateGateState)

      gemv(1, this.weights['W_r'].tensor.transpose(1, 0), currentX.tensor, 1, tempXR.tensor)
      gemv(1, this.weights['U_r'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHR.tensor)
      this._combine(currentResetGateState.tensor, tempXR.tensor, tempHR.tensor, this.weights['b_r'].tensor)
      this.recurrentActivationFunc(currentResetGateState)

      ops.muleq(currentResetGateState.tensor, previousHiddenState.tensor)
      gemv(1, this.weights['W_h'].tensor.transpose(1, 0), currentX.tensor, 1, tempXH.tensor)
      gemv(1, this.weights['U_h'].tensor.transpose(1, 0), currentResetGateState.tensor, 1, tempHH.tensor)
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b_h'].tensor)
      this.activationFunc(currentHiddenState)

      this._update(currentHiddenState.tensor, previousHiddenState.tensor, currentUpdateGateState.tensor)
    }

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i
      ops.assign(currentX.tensor, x.tensor.pick(inputIndex, null))

      // clear temp tensors
      const tempTensors = [tempXZ, tempHZ, tempXR, tempHR, tempXH, tempHH]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))

      // advance timestep
      _step()

      if (this.returnSequences) {
        ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor)
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence
    } else {
      this.output = currentHiddenState
    }

    if (this.stateful) {
      this.currentHiddenState = currentHiddenState
    }
  }

  /**
   * Advance time step in _callGPU
   */
  _stepGPU() {
    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.previousHiddenState,
      inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'source' }]
    })

    // update gate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXZ,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_z'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_z'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_z'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHZ,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_z'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_z'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_z'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentUpdateGateStatePreactiv,
      inputs: [
        { texture: this.tempXZ.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHZ.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_z'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.recurrentActivation !== 'linear') {
      webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentUpdateGateState,
        inputs: [{ texture: this.currentUpdateGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentUpdateGateState = this.currentUpdateGateStatePreactiv
    }

    // reset gate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXR,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_r'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_r'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_r'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHR,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_r'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_r'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_r'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentResetGateStatePreactiv,
      inputs: [
        { texture: this.tempXR.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHR.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_r'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.recurrentActivation !== 'linear') {
      webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentResetGateState,
        inputs: [{ texture: this.currentResetGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentResetGateState = this.currentResetGateStatePreactiv
    }

    // hidden state

    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentResetGateStateCopy,
      inputs: [{ texture: this.currentResetGateState.glTexture, type: '2d', name: 'source' }]
    })

    webgl2.runProgram({
      program: this.gateProductProgram,
      output: this.currentResetGateState,
      inputs: [
        { texture: this.currentResetGateStateCopy.glTexture, type: '2d', name: 't1' },
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 't2' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXH,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_h'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_h'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_h'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHH,
      inputs: [
        { texture: this.currentResetGateState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_h'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_h'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_h'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentHiddenStatePreactiv,
      inputs: [
        { texture: this.tempXH.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHH.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_h'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.activation !== 'linear') {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.currentHiddenState,
        inputs: [{ texture: this.currentHiddenStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentHiddenState = this.currentHiddenStatePreactiv
    }

    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentHiddenStateCopy,
      inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'source' }]
    })

    webgl2.runProgram({
      program: this.updateProgram,
      output: this.currentHiddenState,
      inputs: [
        { texture: this.currentHiddenStateCopy.glTexture, type: '2d', name: 'h' },
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'htm1' },
        { texture: this.currentUpdateGateState.glTexture, type: '2d', name: 'z' }
      ]
    })
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

    const dimUpdateGate = this.weights['b_z'].glTextureShape[1]
    const dimResetGate = this.weights['b_r'].glTextureShape[1]
    const dimHiddenState = this.weights['b_h'].glTextureShape[1]

    if (!this.currentHiddenState || !this.stateful) {
      this.currentHiddenState = new Tensor([], [dimHiddenState])
      this.currentHiddenState.createGLTexture()
    }
    if (!this.currentHiddenStateCopy) {
      this.currentHiddenStateCopy = new Tensor([], [dimHiddenState])
      this.currentHiddenStateCopy.createGLTexture()
    }
    if (!this.currentHiddenStatePreactiv) {
      this.currentHiddenStatePreactiv = new Tensor([], [dimHiddenState])
      this.currentHiddenStatePreactiv.createGLTexture()
    }

    if (!this.currentUpdateGateState) {
      this.currentUpdateGateState = new Tensor([], [dimUpdateGate])
      this.currentUpdateGateState.createGLTexture()
    }
    if (!this.currentUpdateGateStatePreactiv) {
      this.currentUpdateGateStatePreactiv = new Tensor([], [dimUpdateGate])
      this.currentUpdateGateStatePreactiv.createGLTexture()
    }
    if (!this.tempXZ) {
      this.tempXZ = new Tensor([], [dimUpdateGate])
      this.tempXZ.createGLTexture()
    }
    if (!this.tempHZ) {
      this.tempHZ = new Tensor([], [dimUpdateGate])
      this.tempHZ.createGLTexture()
    }

    if (!this.currentResetGateState) {
      this.currentResetGateState = new Tensor([], [dimResetGate])
      this.currentResetGateState.createGLTexture()
    }
    if (!this.currentResetGateStateCopy) {
      this.currentResetGateStateCopy = new Tensor([], [dimResetGate])
      this.currentResetGateStateCopy.createGLTexture()
    }
    if (!this.currentResetGateStatePreactiv) {
      this.currentResetGateStatePreactiv = new Tensor([], [dimResetGate])
      this.currentResetGateStatePreactiv.createGLTexture()
    }
    if (!this.tempXR) {
      this.tempXR = new Tensor([], [dimResetGate])
      this.tempXR.createGLTexture()
    }
    if (!this.tempHR) {
      this.tempHR = new Tensor([], [dimResetGate])
      this.tempHR.createGLTexture()
    }

    if (!this.tempXH) {
      this.tempXH = new Tensor([], [dimHiddenState])
      this.tempXH.createGLTexture()
    }
    if (!this.tempHH) {
      this.tempHH = new Tensor([], [dimHiddenState])
      this.tempHH.createGLTexture()
    }
    if (!this.previousHiddenState) {
      this.previousHiddenState = new Tensor([], [dimHiddenState])
      this.previousHiddenState.createGLTexture()
    }

    if (!this.hiddenStateSequence) {
      this.hiddenStateSequence = new Tensor([], [x.glTextureShape[0], dimHiddenState])
      this.hiddenStateSequence.createGLTexture()
    }
    if (!this.hiddenStateSequenceCopy) {
      this.hiddenStateSequenceCopy = new Tensor([], [x.glTextureShape[0], dimHiddenState])
      this.hiddenStateSequenceCopy.createGLTexture()
    }

    if (!this.currentX) {
      this.currentX = new Tensor([], [x.glTextureShape[1]])
      this.currentX.createGLTexture()
    }

    for (let i = 0, len = x.glTextureShape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i

      webgl2.runProgram({
        program: this.timestepReadProgram,
        output: this.currentX,
        inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }],
        uniforms: [{ value: inputIndex, type: 'int', name: 'index' }]
      })

      this._stepGPU()

      if (this.returnSequences) {
        webgl2.runProgram({
          program: this.copyTextureProgram,
          output: this.hiddenStateSequenceCopy,
          inputs: [{ texture: this.hiddenStateSequence.glTexture, type: '2d', name: 'source' }]
        })
        webgl2.runProgram({
          program: this.timestepWriteProgram,
          output: this.hiddenStateSequence,
          inputs: [
            { texture: this.currentHiddenState.glTexture, type: '2d', name: 'x' },
            { texture: this.hiddenStateSequenceCopy.glTexture, type: '2d', name: 'y' }
          ],
          uniforms: [{ value: i, type: 'int', name: 'index' }]
        })
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence
    } else {
      this.output = this.currentHiddenState
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}
