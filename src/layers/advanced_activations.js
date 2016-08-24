import { Layer } from '../engine/topology'
import { relu } from '../activations'
import cwise from 'cwise'

/**
* LeakyReLU advanced activation layer class
*/
export class LeakyReLU extends Layer {
  /**
  * Creates a LeakyReLU activation layer
  * @param {number} alpha - negative slope coefficient
  */
  constructor (alpha = 0.3) {
    super({})
    this.alpha = alpha
  }

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    relu(x, { alpha: this.alpha })
    return x
  }
}

/**
* PReLU advanced activation layer class
* reference code:
* ```
* pos = K.relu(x)
* neg = self.alphas * (x - abs(x)) * 0.5
* return pos + neg
* ```
*/
export class PReLU extends Layer {
  /**
  * Creates a PReLU activation layer
  */
  constructor () {
    super({})

    /**
    * Layer weights specification
    */
    this.params = ['alphas']
  }

  _compute = cwise({
    args: ['array', 'array'],
    body: function (_x, alpha) {
      _x = Math.max(_x, 0) + alpha * Math.min(_x, 0)
    }
  })

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._compute(x.tensor, this.weights.alphas.tensor)
    return x
  }
}

/**
* ELU advanced activation layer class
*/
export class ELU extends Layer {
  /**
  * Creates a ELU activation layer
  * @param {number} alpha - scale for the negative factor
  */
  constructor (alpha = 1.0) {
    super({})
    this.alpha = alpha
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function (_x, alpha) {
      _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1)
    }
  })

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._compute(x.tensor, this.alpha)
    return x
  }
}

/**
* ParametricSoftplus advanced activation layer class
* alpha * log(1 + exp(beta * X))
*/
export class ParametricSoftplus extends Layer {
  /**
  * Creates a ParametricSoftplus activation layer
  */
  constructor () {
    super({})

    /**
    * Layer weights specification
    */
    this.params = ['alphas', 'betas']
  }

  _compute = cwise({
    args: ['array', 'array', 'array'],
    body: function (_x, alpha, beta) {
      _x = alpha * Math.log(1 + Math.exp(beta * _x))
    }
  })

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._compute(x.tensor, this.weights.alphas.tensor, this.weights.betas.tensor)
    return x
  }
}

/**
* ThresholdedReLU advanced activation layer class
*/
export class ThresholdedReLU extends Layer {
  /**
  * Creates a ThresholdedReLU activation layer
  * @param {number} theta - float >= 0. Threshold location of activation.
  */
  constructor (theta = 1.0) {
    super({})
    this.theta = theta
  }

  _compute = cwise({
    args: ['array', 'scalar'],
    body: function (_x, theta) {
      _x = _x * Number(_x > theta)
    }
  })

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._compute(x.tensor, this.theta)
    return x
  }
}

/**
* SReLU advanced activation layer class
* S-shaped Rectified Linear Unit
*/
export class SReLU extends Layer {
  /**
  * Creates a SReLU activation layer
  */
  constructor () {
    super({})

    /**
    * Layer weights specification
    */
    this.params = ['t_left', 'a_left', 't_right', 'a_right']
  }

  // t_right_actual = t_left + abs(t_right)
  // Y_left_and_center = t_left + K.relu(x - t_left, a_left, t_right_actual - t_left)
  // Y_right = K.relu(x - t_right_actual) * a_right
  // return Y_left_and_center + Y_right
  _compute = cwise({
    args: ['array', 'array', 'array', 'array', 'array'],
    body: function (_x, tL, aL, tR, aR) {
      _x = tL + Math.min(Math.max(_x - tL, 0), Math.abs(tR)) + aL * Math.min(_x - tL, 0) +
        Math.max(_x - (tL + Math.abs(tR)), 0) * aR
    }
  })

  /**
  * Method for layer computational logic
  * @param {Tensor} x
  * @returns {Tensor} x
  */
  call = x => {
    this._compute(
      x.tensor,
      this.weights.t_left.tensor,
      this.weights.a_left.tensor,
      this.weights.t_right.tensor,
      this.weights.a_right.tensor
    )
    return x
  }
}
