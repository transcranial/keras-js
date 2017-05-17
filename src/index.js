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

export { Model, Tensor, activations, layers, testUtils }
