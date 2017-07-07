describe('core layer: Activation', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%ccore layer: Activation', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    it('[core.Activation.0] [CPU] should produce expected values for tanh activation following Dense layer', function() {
      const key = 'core.Activation.0'
      console.log(`\n%c[${key}] [CPU] test 1 (tanh)`, styles.h3)
      let testLayer1 = new layers.Dense({ units: 2 })
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation({ activation: 'tanh' })
      const startTime = performance.now()
      t = testLayer2.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[core.Activation.1] [CPU] should produce expected values for hard_sigmoid activation following Dense layer', function() {
      const key = 'core.Activation.1'
      console.log(`\n%c[${key}] [CPU] test 2 (hard_sigmoid)`, styles.h3)
      let testLayer1 = new layers.Dense({ units: 2 })
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation({ activation: 'hard_sigmoid' })
      const startTime = performance.now()
      t = testLayer2.call(t)
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
  * GPU
  *********************************************************/
  describe('GPU', function() {
    before(function() {
      console.log('\n%cGPU', styles.h2)
    })

    it('[core.Activation.0] [GPU] should produce expected values for tanh activation following Dense layer', function() {
      const key = 'core.Activation.0'
      console.log(`\n%c[${key}] [GPU] test 1 (tanh)`, styles.h3)
      let testLayer1 = new layers.Dense({ units: 2, gpu: true })
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation({ activation: 'tanh', gpu: true })
      const startTime = performance.now()
      t = testLayer2.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[core.Activation.1] [GPU] should produce expected values for hard_sigmoid activation following Dense layer', function() {
      const key = 'core.Activation.1'
      console.log(`\n%c[${key}] [GPU] test 2 (hard_sigmoid)`, styles.h3)
      let testLayer1 = new layers.Dense({ units: 2, gpu: true })
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation({ activation: 'hard_sigmoid', gpu: true })
      const startTime = performance.now()
      t = testLayer2.call(t)
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
