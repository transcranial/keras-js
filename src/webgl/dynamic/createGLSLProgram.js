import add from './merge/add'
import average from './merge/average'
import concatenate from './merge/concatenate'
import maximum from './merge/maximum'
import minimum from './merge/minimum'
import multiply from './merge/multiply'
import subtract from './merge/subtract'

export default function createGLSLProgram(program, ...args) {
  switch (program) {
    // merge
    case 'add':
      return add(...args)
    case 'average':
      return average(...args)
    case 'concatenate':
      return concatenate(...args)
    case 'maximum':
      return maximum(...args)
    case 'minimum':
      return minimum(...args)
    case 'multiply':
      return multiply(...args)
    case 'subtract':
      return subtract(...args)

    default:
      throw new Error('GLSL program not found')
  }
}
