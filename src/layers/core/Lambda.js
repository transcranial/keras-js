import Layer from '../../Layer'

/**
 * Lambda layer class
 * This layer requires you to re-implement lambda nodes in javascript,
 * as we do not have a python runtime available.
 */
export default class Lambda extends Layer {

  /**
   * Creates a Lambda layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Lambda';

    if(this.functions[attrs.name]) {
      this._call = this.functions[attrs.name].bind(this);
    } else {
      console.log("Missing lambda, using No_op! Implement it by defining require(\"keras-js/layers/core/Lambda\").functions["+attrs.name+"]");
      this._call = x => ({output: x, inputShape: x.tensor.shape});
    }

    if(this.initializers[attrs.name]) {
      this.initializers[attrs.name].bind(this)();
    } else {
      console.log("No lambda initializer. Disable this warning by defining require(\"keras-js/layers/core/Lambda\").initializers["+attrs.name+"]");
    }
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) { 
    return Object.assign(this, this._call(x)).output;
  }

}
Lambda.prototype.functions = {};
Lambda.prototype.initializers = {}

exports.default = Lambda;
