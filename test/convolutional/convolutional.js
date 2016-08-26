/* eslint-env browser, mocha */

describe('Layers: Convolutional', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function () {
    console.log('\n%cLayers: Convolutional', styles.h1)
  })

  /*********************************************************
  * Convolution2D
  *********************************************************/

  describe('Convolution2D', function () {
    before(function () {
      console.log('\n%cConvolution2D', styles.h2)
    })

    const testParams = [
      {
        mode: 'CPU',
        inputShape: [5, 5, 2],
        kernelShape: [4, 3, 3],
        attrs: { activation: 'linear', borderMode: 'valid', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        mode: 'CPU',
        inputShape: [5, 5, 2],
        kernelShape: [4, 3, 3],
        attrs: { activation: 'linear', borderMode: 'valid', subsample: [1, 1], dimOrdering: 'tf', bias: false }
      },
      {
        mode: 'CPU',
        inputShape: [5, 5, 2],
        kernelShape: [4, 3, 3],
        attrs: { activation: 'relu', borderMode: 'valid', subsample: [2, 2], dimOrdering: 'tf', bias: true }
      },
      {
        mode: 'CPU',
        inputShape: [7, 7, 3],
        kernelShape: [5, 4, 4],
        attrs: { activation: 'relu', borderMode: 'valid', subsample: [2, 1], dimOrdering: 'tf', bias: true }
      },
      {
        mode: 'CPU',
        inputShape: [5, 5, 2],
        kernelShape: [4, 3, 3],
        attrs: { activation: 'relu', borderMode: 'same', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        mode: 'CPU',
        inputShape: [4, 4, 2],
        kernelShape: [4, 3, 3],
        attrs: { activation: 'relu', borderMode: 'same', subsample: [2, 2], dimOrdering: 'tf', bias: true }
      }
    ]

    testParams.forEach(({ mode, inputShape, kernelShape, attrs }, i) => {
      const key = `convolutional.Convolution2D.${i}`
      const [inputRows, inputCols, inputChannels] = inputShape
      const [nbFilter, nbRow, nbCol] = kernelShape
      const title = `[${key}] [CPU] test 1: ${nbFilter} ${nbRow}x${nbCol} filters on ${inputRows}x${inputCols}x${inputChannels} input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsample=${attrs.subsample}, dim_ordering='${attrs.dimOrdering}', bias=${attrs.bias}`

      it(title, function () {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.Convolution2D(nbFilter, nbRow, nbCol, attrs)
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        const startTime = performance.now()
        t = testLayer.call(t)
        const endTime = performance.now()
        console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
        logTime(startTime, endTime)
        const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
        const shapeExpected = TEST_DATA[key].expected.shape
        assert.deepEqual(t.tensor.shape, shapeExpected)
        assert.isTrue(approxEquals(t.tensor, dataExpected))
      })
    })
  })
})
