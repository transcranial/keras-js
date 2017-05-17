describe('advanced activation layers', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cadvanced activation layers', styles.h1)
  })

  /*********************************************************
  * LeakyReLU
  *********************************************************/
  describe('LeakyReLU', function() {
    before(function() {
      console.log('\n%cLeakyReLU', styles.h2)
    })

    it('[advanced_activations.LeakyReLU.0] should produce expected values', function() {
      const key = 'advanced_activations.LeakyReLU.0'
      console.log(`\n%c[${key}] alpha=0.4`, styles.h3)
      let testLayer = new layers.LeakyReLU({ alpha: 0.4 })
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

  /*********************************************************
  * PReLU
  *********************************************************/
  describe('PReLU', function() {
    before(function() {
      console.log('\n%cPReLU', styles.h2)
    })

    it('[advanced_activations.PReLU.0] should produce expected values', function() {
      const key = 'advanced_activations.PReLU.0'
      console.log(`\n%c[${key}] weights: alphas`, styles.h3)
      let testLayer = new layers.PReLU()
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

  /*********************************************************
  * ELU
  *********************************************************/
  describe('ELU', function() {
    before(function() {
      console.log('\n%cELU', styles.h2)
    })

    it('[advanced_activations.ELU.0] should produce expected values', function() {
      const key = 'advanced_activations.ELU.0'
      console.log(`\n%c[${key}] alpha=1.1`, styles.h3)
      let testLayer = new layers.ELU({ alpha: 1.1 })
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

  /*********************************************************
  * ThresholdedReLU
  *********************************************************/
  describe('ThresholdedReLU', function() {
    before(function() {
      console.log('\n%cThresholdedReLU', styles.h2)
    })

    it('[advanced_activations.ThresholdedReLU.0] should produce expected values', function() {
      const key = 'advanced_activations.ThresholdedReLU.0'
      console.log(`\n%c[${key}] theta=0.9`, styles.h3)
      let testLayer = new layers.ThresholdedReLU({ theta: 0.9 })
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
