import _Merge from './_Merge'

/**
 * Average merge layer class, extends abstract _Merge class
 */
export default class Average extends _Merge {
  /**
   * Creates a Average merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Average'

    this.mode = 'ave'
  }
}
