import ops from 'ndarray-ops'

/**
 * In-place operation: softmax activation function
 *
 * @param {Tensor} x
 */
export default function softmax(x) {
  if (x.tensor.shape.length === 1) {
    const maxval = ops.sup(x.tensor)
    ops.subseq(x.tensor, maxval)
    ops.expeq(x.tensor)
    const sum = ops.sum(x.tensor)
    ops.divseq(x.tensor, sum)
  } else if (x.tensor.shape.length === 2) {
    for (let i = 0; i < x.tensor.shape[0]; i++) {
      const maxval = ops.sup(x.tensor.pick(i, null))
      ops.subseq(x.tensor.pick(i, null), maxval)
      ops.expeq(x.tensor.pick(i, null))
      const sum = ops.sum(x.tensor.pick(i, null))
      ops.divseq(x.tensor.pick(i, null), sum)
    }
  } else if (x.tensor.shape.length === 3) {
    for (let i=0; i< x.tensor.shape[0]; i++){
      for(let j=0;j<x.tensor.shape[1]; j++){
        const maxval = _ndarrayOps.default.sup(x.tensor.pick(i,j,null));
        _ndarrayOps.default.subseq(x.tensor.pick(i,j, null), maxval);
        _ndarrayOps.default.expeq(x.tensor.pick(i,j, null));
        const sum = _ndarrayOps.default.sum(x.tensor.pick(i,j, null));
        _ndarrayOps.default.divseq(x.tensor.pick(i,j, null), sum);
      }
    }
  } else {
    throw new Error(`[activations.softmax] tensor shape ${x.tensor.shape} not supported.`)
  }
}
