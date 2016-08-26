import * as activations from '../../activations'
import Layer from '../../engine/Layer'

/**
* Activation layer class
*/
export default class Activation extends Layer {
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
