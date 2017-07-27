describe('merge layer: Maximum', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cmerge layer: Maximum', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    it('[merge.Maximum.0] [CPU] should produce expected values', function() {
      const key = 'merge.Maximum.0'
      console.log(`\n%c[${key}] [CPU]`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2 = new layers.Maximum()
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed(t1a.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(t1b.tensor))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })
  })

  /*********************************************************
  * GPU
  *********************************************************/
  describe('GPU', function() {
    before(function() {
      console.log('\n%cGPU', styles.h2)
    })

    it('[merge.Maximum.0] [GPU] should produce expected values', function() {
      const key = 'merge.Maximum.0'
      console.log(`\n%c[${key}] [GPU]`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2, gpu: true })
      let testLayer1b = new layers.Dense({ units: 2, gpu: true })
      let testLayer2 = new layers.Maximum({ gpu: true })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed(t1a.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(t1b.tensor))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })
  })
})
