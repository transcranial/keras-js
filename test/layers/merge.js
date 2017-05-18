describe('merge layers', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cmerge layers', styles.h1)
  })

  /*********************************************************
  * Add
  *********************************************************/
  describe('Add', function() {
    before(function() {
      console.log('\n%cAdd', styles.h2)
    })

    it('[merge.Add.0] should produce expected values', function() {
      const key = 'merge.Add.0'
      console.log(`\n%c[${key}]`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2 = new layers.Add()
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
  * Multiply
  *********************************************************/
  describe('Multiply', function() {
    before(function() {
      console.log('\n%cMultiply', styles.h2)
    })

    it('[merge.Multiply.0] should produce expected values', function() {
      const key = 'merge.Multiply.0'
      console.log(`\n%c[${key}]`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2 = new layers.Multiply()
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
  * Average
  *********************************************************/
  describe('Average', function() {
    before(function() {
      console.log('\n%cAverage', styles.h2)
    })

    it('[merge.Average.0] should produce expected values', function() {
      const key = 'merge.Average.0'
      console.log(`\n%c[${key}]`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2 = new layers.Average()
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
  * Maximum
  *********************************************************/
  describe('Maximum', function() {
    before(function() {
      console.log('\n%cMaximum', styles.h2)
    })

    it('[merge.Maximum.0] should produce expected values', function() {
      const key = 'merge.Maximum.0'
      console.log(`\n%c[${key}]`, styles.h3)
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
  * Concatenate
  *********************************************************/
  describe('Concatenate', function() {
    before(function() {
      console.log('\n%cConcatenate', styles.h2)
    })

    it('[merge.Concatenate.0] should produce expected values: 1D inputs, axis=-1', function() {
      const key = 'merge.Concatenate.0'
      console.log(`\n%c[${key}] 1D inputs, axis=-1`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2 = new layers.Concatenate({ axis: -1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      tb = testLayer1b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer2.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Concatenate.1] should produce expected values: 2D inputs, axis=-1', function() {
      const key = 'merge.Concatenate.1'
      console.log(`\n%c[${key}] 2D inputs, axis=-1`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Concatenate({ axis: -1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Concatenate.2] should produce expected values: 2D inputs, axis=-2', function() {
      const key = 'merge.Concatenate.2'
      console.log(`\n%c[${key}] 2D inputs, axis=-2`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Concatenate({ axis: -2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Concatenate.3] should produce expected values: 2D inputs, axis=1', function() {
      const key = 'merge.Concatenate.3'
      console.log(`\n%c[${key}] 2D inputs, axis=1`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Concatenate({ axis: 1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Concatenate.4] should produce expected values: 2D inputs, axis=2', function() {
      const key = 'merge.Concatenate.4'
      console.log(`\n%c[${key}] 2D inputs, axis=2`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Concatenate({ axis: 2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
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
  * Dot
  *********************************************************/
  describe('Dot', function() {
    before(function() {
      console.log('\n%cDot', styles.h2)
    })

    it('[merge.Dot.0] should produce expected values: 2D x 2D inputs, axes=1, normalize=False', function() {
      const key = 'merge.Dot.0'
      console.log(`\n%c[${key}] 2D x 2D inputs, axes=1, normalize=False`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Dot({ axes: 1, normalize: false })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Dot.1] should produce expected values: 2D x 2D inputs, axes=2, normalize=False', function() {
      const key = 'merge.Dot.1'
      console.log(`\n%c[${key}] 2D x 2D inputs, axes=2, normalize=False`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Dot({ axes: 2, normalize: false })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Dot.2] should produce expected values: 2D x 2D inputs, axes=1, normalize=True', function() {
      const key = 'merge.Dot.2'
      console.log(`\n%c[${key}] 2D x 2D inputs, axes=1, normalize=True`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Dot({ axes: 1, normalize: true })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Dot.3] should produce expected values: 2D x 2D inputs, axes=2, normalize=True', function() {
      const key = 'merge.Dot.3'
      console.log(`\n%c[${key}] 2D x 2D inputs, axes=2, normalize=True`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Dot({ axes: 2, normalize: true })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[merge.Dot.4] should produce expected values: 2D x 2D inputs, axes=(2,2), normalize=True', function() {
      const key = 'merge.Dot.4'
      console.log(`\n%c[${key}] 2D x 2D inputs, axes=(2,2), normalize=True`, styles.h3)
      let testLayer1a = new layers.Dense({ units: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ units: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Dot({ axes: [2, 2], normalize: true })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed(ta.tensor))
      console.log('%cin', styles.h4, stringifyCondensed(tb.tensor))
      const startTime = performance.now()
      let t = testLayer3.call([ta, tb])
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
