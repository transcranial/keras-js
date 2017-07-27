/* global nnpack */
import Promise from 'bluebird'
import * as enums from './enums'

class NNPACK {
  constructor() {
    this.initialized = false
    this.initialize()
    this.threadpool = null
  }

  async initialize() {
    // time out in 10s
    for (let i = 0; i < 10000; i++) {
      if (!nnpack.usingWasm) {
        await Promise.delay(1)
      } else {
        console.log(`[NNPACK WebAssembly module] created in ${i} ms.`)
        break
      }
    }

    if (!nnpack.usingWasm) {
      this.initialized = false
      throw new Error('[NNPACK WebAssembly module] failed to create module.')
    }

    //NOTE: no support for multiple threads in WebAssembly yet
    this.threadpool = nnpack.ccall('pthreadpool_create', 'number', ['number'], [0])

    const status = nnpack.ccall('nnp_initialize')
    if (status !== enums.NNP_STATUS.SUCCESS) {
      throw new Error(`[NNPACK WebAssembly module] initialization failed: error code ${status}.`)
    }

    this.initialized = true
  }

  deinitialize() {
    nnpack.ccall('nnp_deinitialize')
  }

  /**
   * wrapper for nnp_convolution_inference
   */
  convolutionInference() {}

  /**
   * wrapper for nnp_fully_connected_inference
   */
  fullyConnectedInference() {}

  /**
   * wrapper for nnp_max_pooling_output
   */
  maxPoolingOutput() {}

  /**
   * wrapper for nnp_softmax_output
   */
  softmaxOutput() {}

  /**
   * wrapper for nnp_relu_output
   */
  reluOutput(x, opts = {}) {
    const { alpha = 0 } = opts
    const status = nnpack.ccall(
      'nnp_relu_output',
      'number',
      ['number', 'number', 'array', 'array', 'number', 'number'],
      [1, channels, input, output, alpha, this.threadpool]
    )
    if (status !== enums.NNP_STATUS.SUCCESS) {
      console.log(`[NNPACK WebAssembly module] nnp_relu_output error`)
    }
  }
}

const nnpackInstance = new NNPACK()

export default nnpackInstance
