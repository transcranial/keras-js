import _Merge from './_Merge'

/**
 * Add merge layer class, extends abstract _Merge class
 */
export default class Add extends _Merge {
  /**
   * Creates a Add merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Add'

    this.mode = 'sum'
  }
}
