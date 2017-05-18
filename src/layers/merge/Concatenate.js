import _Merge from './_Merge'

/**
 * Concatenate merge layer class, extends abstract _Merge class
 */
export default class Concatenate extends _Merge {
  /**
   * Creates a Concatenate merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Concatenate'

    this.mode = 'concat'

    const { axis = -1 } = attrs

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = axis <= 0 ? axis : axis - 1
  }
}
