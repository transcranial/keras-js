import 'babel-polyfill'
import Model from './Model'
import Tensor from './Tensor'
import * as activations from './activations'
import * as layers from './layers'
import * as testUtils from './utils/testUtils'

if (typeof window !== 'undefined') {
  const weblas = require('weblas/dist/weblas')
  window.weblas = weblas
}

/*
if (typeof window !== 'undefined' && 'WebAssembly' in window) {
  window.nnpack = {
    wasmBinary: require('arraybuffer-loader!./nnpack/libnnpack.wasm')
  }
  // libnnpack.js replaces first line so that we can use as global:
  // `var Modules;`
  // with
  // `var Modules = window.nnpack;`
  require('script-loader!./nnpack/libnnpack.js')
}
*/

export { Model, Tensor, activations, layers, testUtils }
