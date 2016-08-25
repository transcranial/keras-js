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

    it('[convolutional.Convolution2D.0] [CPU] should produce expected values for activation=linear, borderMode=valid, subsample=[1,1], dimOrdering=tf, biase=true', function () {
      const key = 'convolutional.Convolution2D.0'
      const [nbRow, nbCol, nbFilter] = TEST_DATA[key].expected.shape
      const attrs = { activation: 'linear', borderMode: 'valid', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      console.log(`\n%c[${key}] [CPU] test 1: ${nbFilter} ${nbRow}x${nbCol} filters on 5x5x2 input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsample=${attrs.subsample}, dim_ordering='${attrs.dimOrdering}', bias=${attrs.bias}`, styles.h3)
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
