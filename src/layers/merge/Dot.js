import _Merge from './_Merge'

/**
 * Dot merge layer class, extends abstract _Merge class
 */
export default class Dot extends _Merge {
  /**
   * Creates a Dot merge layer
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'Dot'

    this.mode = 'dot'

    const { axes = -1, normalize = false } = attrs

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    if (Array.isArray(axes)) {
      this.dotAxes = [axes[0] <= 0 ? axes[0] : axes[0] - 1, axes[1] <= 0 ? axes[1] : axes[1] - 1]
    } else {
      this.dotAxes = [axes <= 0 ? axes : axes - 1, axes <= 0 ? axes : axes - 1]
    }

    this.normalize = normalize
  }
}
