describe('normalization layer: BatchNormalization', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { attrs: { epsilon: 0.00001, axis: -1, center: true, scale: true } },
    { attrs: { epsilon: 0.01, axis: -1, center: true, scale: true } },
    { attrs: { epsilon: 0.00001, axis: 1, center: true, scale: true } },
    { attrs: { epsilon: 0.00001, axis: 2, center: true, scale: true } },
    { attrs: { epsilon: 0.00001, axis: 3, center: true, scale: false } },
    { attrs: { epsilon: 0.00001, axis: -1, center: false, scale: true } },
    { attrs: { epsilon: 0.001, axis: -1, center: false, scale: false } }
  ]

  before(function() {
    console.log('\n%cnormalization layer: BatchNormalization', styles.h1)
  })

  testParams.forEach(({ attrs }, i) => {
    const key = `normalization.BatchNormalization.${i}`
    const title = `[${key}] test: epsilon='${attrs.epsilon}', axis=${attrs.axis}, center=${attrs.center}, scale=${attrs.scale}`

    it(title, function() {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.BatchNormalization(attrs)
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
