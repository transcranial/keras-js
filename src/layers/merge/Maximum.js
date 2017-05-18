import _Merge from './_Merge'

/**
 * Maximum merge layer class, extends abstract _Merge class
 */
export default class Maximum extends _Merge {
  /**
   * Creates a Maximum merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Maximum'

    this.mode = 'max'
  }
}
