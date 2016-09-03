/* eslint-env browser, mocha */

describe('pooling layer: GlobalMaxPooling2D', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [6, 6, 3],
      attrs: { dimOrdering: 'tf' }
    },
    {
      inputShape: [3, 6, 6],
      attrs: { dimOrdering: 'th' }
    },
    {
      inputShape: [5, 3, 2],
      attrs: { dimOrdering: 'tf' }
    }
  ]

  before(function () {
    console.log('\n%cpooling layer: GlobalMaxPooling2D', styles.h1)
  })

  testParams.forEach(({ inputShape, attrs }, i) => {
    const key = `pooling.GlobalMaxPooling2D.${i}`
    const [inputRows, inputCols, inputChannels] = inputShape
    const title = `[${key}] test: ${inputRows}x${inputCols}x${inputChannels} input, dimOrdering=${attrs.dimOrdering}`

    it(title, function () {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.GlobalMaxPooling2D(attrs)
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
