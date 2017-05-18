import _Merge from './_Merge'

/**
 * Multiply merge layer class, extends abstract _Merge class
 */
export default class Multiply extends _Merge {
  /**
   * Creates a Multiply merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Multiply'

    this.mode = 'mul'
  }
}
