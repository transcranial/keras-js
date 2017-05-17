import unpack from 'ndarray-unpack'
import flattenDeep from 'lodash/flattenDeep'
import isFinite from 'lodash/isFinite'

/**
 * Compares an ndarray's data element-wise to dataExpected,
 * within a certain tolerance. We unpack the ndarray first since
 * stride/offset prevents us from comparing the array data
 * element-wise directly.
 */
export function approxEquals(ndarrayOut, dataExpected, tol = 0.0001) {
  const a = flattenDeep(unpack(ndarrayOut))
  const b = dataExpected
  if (a.length !== b.length) {
    return false
  }
  for (let i = 0; i < a.length; i++) {
    if (!isFinite(a[i])) {
      return false
    }
    if (a[i] < b[i] - tol || a[i] > b[i] + tol) {
      return false
    }
  }
  return true
}
