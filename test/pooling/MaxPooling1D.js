/* eslint-env browser, mocha */

describe('pooling layer: MaxPooling1D', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: null, borderMode: 'valid' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: 1, borderMode: 'valid' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: 3, borderMode: 'valid' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: null, borderMode: 'same' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: 1, borderMode: 'same' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 2, stride: 3, borderMode: 'same' }
    },
    {
      inputShape: [6, 6],
      attrs: { poolLength: 3, stride: null, borderMode: 'valid' }
    },
    {
      inputShape: [7, 7],
      attrs: { poolLength: 3, stride: 1, borderMode: 'same' }
    },
    {
      inputShape: [7, 7],
      attrs: { poolLength: 3, stride: 3, borderMode: 'same' }
    }
  ]

  before(function () {
    console.log('\n%cpooling layer: MaxPooling1D', styles.h1)
  })

  testParams.forEach(({ inputShape, attrs }, i) => {
    const key = `pooling.MaxPooling1D.${i}`
    const [inputLength, inputFeatures] = inputShape
    const title = `[${key}] test: ${inputLength}x${inputFeatures} input, poolLength='${attrs.poolLength}', stride=${attrs.stride}, borderMode=${attrs.borderMode}`

    it(title, function () {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.MaxPooling1D(attrs)
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
