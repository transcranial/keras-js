describe('pooling layer: GlobalMaxPooling3D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { inputShape: [6, 6, 3, 4], attrs: { data_format: 'channels_last' } },
    { inputShape: [3, 6, 6, 3], attrs: { data_format: 'channels_first' } },
    { inputShape: [5, 3, 2, 1], attrs: { data_format: 'channels_last' } }
  ]

  before(function() {
    console.log('\n%cpooling layer: GlobalMaxPooling3D', styles.h1)
  })

  testParams.forEach(({ inputShape, attrs }, i) => {
    const key = `pooling.GlobalMaxPooling3D.${i}`
    const title = `[${key}] test: ${inputShape} input, data_format=${attrs.data_format}`

    it(title, function() {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.GlobalMaxPooling3D(attrs)
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
