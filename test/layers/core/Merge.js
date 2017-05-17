describe('core layer: Merge', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%ccore layer: Merge', styles.h1)
  })

  /*********************************************************
  * sum
  *********************************************************/
  describe('sum', function() {
    before(function() {
      console.log('\n%csum', styles.h2)
    })

    it('[core.Merge.0] should produce expected values in sum mode', function() {
      const key = 'core.Merge.0'
      console.log(`\n%c[${key}] mode: sum`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2 = new layers.Merge({ mode: 'sum' })
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
  * mul
  *********************************************************/
  describe('mul', function() {
    before(function() {
      console.log('\n%cmul', styles.h2)
    })

    it('[core.Merge.1] should produce expected values in mul mode', function() {
      const key = 'core.Merge.1'
      console.log(`\n%c[${key}] mode: mul`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2 = new layers.Merge({ mode: 'mul' })
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
  * ave
  *********************************************************/
  describe('ave', function() {
    before(function() {
      console.log('\n%cave', styles.h2)
    })

    it('[core.Merge.2] should produce expected values in ave mode', function() {
      const key = 'core.Merge.2'
      console.log(`\n%c[${key}] mode: ave`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2 = new layers.Merge({ mode: 'ave' })
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
  * max
  *********************************************************/
  describe('max', function() {
    before(function() {
      console.log('\n%cmax', styles.h2)
    })

    it('[core.Merge.3] should produce expected values in max mode', function() {
      const key = 'core.Merge.3'
      console.log(`\n%c[${key}] mode: max`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2 = new layers.Merge({ mode: 'max' })
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
  * concat
  *********************************************************/
  describe('concat', function() {
    before(function() {
      console.log('\n%cconcat', styles.h2)
    })

    it('[core.Merge.4] should produce expected values in concat mode (1D)', function() {
      const key = 'core.Merge.4'
      console.log(`\n%c[${key}] mode: concat (1D)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2 = new layers.Merge({ mode: 'concat', concatAxis: -1 })
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

    it('[core.Merge.5] should produce expected values in concat mode (2D, concatAxis=-1)', function() {
      const key = 'core.Merge.5'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=-1)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: -1 })
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

    it('[core.Merge.6] should produce expected values in concat mode (2D, concatAxis=-2)', function() {
      const key = 'core.Merge.6'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=-2)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: -2 })
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

    it('[core.Merge.7] should produce expected values in concat mode (2D, concatAxis=1)', function() {
      const key = 'core.Merge.7'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=1)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: 1 })
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

    it('[core.Merge.8] should produce expected values in concat mode (2D, concatAxis=2)', function() {
      const key = 'core.Merge.8'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=2)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: 2 })
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
  * dot
  *********************************************************/
  describe('dot', function() {
    before(function() {
      console.log('\n%cdot', styles.h2)
    })

    it('[core.Merge.9] should produce expected values in dot mode (2D x 2D, dotAxes=1)', function() {
      const key = 'core.Merge.9'
      console.log(`\n%c[${key}] mode: dot (2D x 2D, dotAxes=1)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'dot', dotAxes: 1 })
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

    it('[core.Merge.10] should produce expected values in dot mode (2D x 2D, dotAxes=2)', function() {
      const key = 'core.Merge.10'
      console.log(`\n%c[${key}] mode: dot (2D x 2D, dotAxes=2)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'dot', dotAxes: 2 })
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
  * cos
  *********************************************************/
  describe('cos', function() {
    before(function() {
      console.log('\n%ccos', styles.h2)
    })

    it('[core.Merge.11] should produce expected values in cos mode (2D x 2D, dotAxes=1)', function() {
      const key = 'core.Merge.11'
      console.log(`\n%c[${key}] mode: cos (2D x 2D, dotAxes=1)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'cos', dotAxes: 1 })
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

    it('[core.Merge.12] should produce expected values in cos mode (2D x 2D, dotAxes=2)', function() {
      const key = 'core.Merge.12'
      console.log(`\n%c[${key}] mode: cos (2D x 2D, dotAxes=2)`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'cos', dotAxes: 2 })
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

    it('[core.Merge.13] should produce expected values in cos mode (2D x 2D, dotAxes=(2,2))', function() {
      const key = 'core.Merge.12'
      console.log(`\n%c[${key}] mode: cos (2D x 2D, dotAxes=(2,2))`, styles.h3)
      let testLayer1a = new layers.Dense({ outputDim: 2 })
      let testLayer2a = new layers.RepeatVector({ n: 3 })
      let testLayer1b = new layers.Dense({ outputDim: 2 })
      let testLayer2b = new layers.RepeatVector({ n: 3 })
      let testLayer3 = new layers.Merge({ mode: 'cos', dotAxes: [2, 2] })
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
