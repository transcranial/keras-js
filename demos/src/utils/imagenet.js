import _ from 'lodash'
import { imagenetClasses } from '../data/imagenet'

/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities, k = 5) {
  const probs = _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities

  const sorted = _.reverse(_.sortBy(probs.map((prob, index) => [prob, index]), probIndex => probIndex[0]))

  const topK = _.take(sorted, k).map(probIndex => {
    const iClass = imagenetClasses[probIndex[1]]
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/, ' '),
      probability: probIndex[0]
    }
  })
  return topK
}
