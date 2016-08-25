/* eslint-env browser, mocha */

describe('Layers: Core', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function () {
    console.log('\n%cLayers: Core', styles.h1)
  })

  /*********************************************************
  * Dense
  *********************************************************/

  describe('Dense', function () {
    before(function () {
      console.log('\n%cDense', styles.h2)
    })

    it('[core.Dense.0] [CPU] should produce expected values', function () {
      const key = 'core.Dense.0'
      console.log(`\n%c[${key}] [CPU] test 1`, styles.h3)
      let testLayer = new layers.Dense(2)
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

    it('[core.Dense.1] [CPU] should produce expected values, with sigmoid activation function', function () {
      const key = 'core.Dense.1'
      console.log(`\n%c[${key}] [CPU] test 2 (with sigmoid activation)`, styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'sigmoid' })
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

    it('[core.Dense.2] [CPU] should produce expected values, with softplus activation function and no bias', function () {
      const key = 'core.Dense.2'
      console.log(`\n%c[${key}] [CPU] test 3 (with softplus activation and no bias)`, styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'softplus', bias: false })
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

    it('[core.Dense.3] [GPU] should produce expected values', function () {
      const key = 'core.Dense.3'
      console.log(`\n%c[${key}] [GPU] test 1`, styles.h3)
      let testLayer = new layers.Dense(2)
      testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape, { useWeblas: true })
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

    it('[core.Dense.4] [GPU] should produce expected values, with sigmoid activation function', function () {
      const key = 'core.Dense.4'
      console.log(`\n%c[${key}] [GPU] test 2 (with sigmoid activation)`, styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'sigmoid' })
      testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape, { useWeblas: true })
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

    it('[core.Dense.5] [GPU] should produce expected values, with softplus activation function and no bias', function () {
      const key = 'core.Dense.5'
      console.log(`\n%c[${key}] [GPU] test 3 (with softplus activation and no bias)`, styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'softplus', bias: false })
      testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape, { useWeblas: true })
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
  * Activation
  *********************************************************/

  describe('Activation', function () {
    before(function () {
      console.log('\n%cActivation', styles.h2)
    })

    it('[core.Activation.0] should produce expected values for tanh activation following Dense layer', function () {
      const key = 'core.Activation.0'
      console.log(`\n%c[${key}] test 1 (tanh)`, styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation('tanh')
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

    it('[core.Activation.1] should produce expected values for hardSigmoid activation following Dense layer', function () {
      const key = 'core.Activation.1'
      console.log(`\n%c[${key}] test 2 (hardSigmoid)`, styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation('hardSigmoid')
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
  * Dropout
  *********************************************************/

  describe('Dropout', function () {
    before(function () {
      console.log('\n%cDropout', styles.h2)
    })

    it('[core.Dropout.0] should just pass through tensor during test time', function () {
      const key = 'core.Dropout.0'
      console.log(`\n%c[${key}] should pass through`, styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Dropout(0.5)
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
  * Flatten
  *********************************************************/

  describe('Flatten', function () {
    before(function () {
      console.log('\n%cFlatten', styles.h2)
    })

    it('[core.Flatten.0] should do nothing for 1D', function () {
      const key = 'core.Flatten.0'
      console.log(`\n%c[${key}] 1D`, styles.h3)
      let testLayer = new layers.Flatten()
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

    it('[core.Flatten.1] should flatten 2D', function () {
      const key = 'core.Flatten.1'
      console.log(`\n%c[${key}] 2D`, styles.h3)
      let testLayer = new layers.Flatten()
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

    it('[core.Flatten.2] should flatten 3D', function () {
      const key = 'core.Flatten.2'
      console.log(`\n%c[${key}] 3D`, styles.h3)
      let testLayer = new layers.Flatten()
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
  * Reshape
  *********************************************************/

  describe('Reshape', function () {
    before(function () {
      console.log('\n%cReshape', styles.h2)
    })

    it('[core.Reshape.0] should be able to go from shape [6] -> [2, 3]', function () {
      const key = 'core.Reshape.0'
      console.log(`\n%c[${key}] shape [6] -> [2, 3]`, styles.h3)
      let testLayer = new layers.Reshape([2, 3])
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

    it('[core.Reshape.1] should be able to go from shape [3, 2] -> [6]', function () {
      const key = 'core.Reshape.1'
      console.log(`\n%c[${key}] shape [3, 2] -> [6]`, styles.h3)
      let testLayer = new layers.Reshape([6])
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

    it('[core.Reshape.2] should be able to go from shape [3, 2, 2] -> [4, 3]', function () {
      const key = 'core.Reshape.2'
      console.log(`\n%c[${key}] shape [3, 2, 2] -> [4, 3]`, styles.h3)
      let testLayer = new layers.Reshape([4, 3])
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
  * Permute
  *********************************************************/

  describe('Permute', function () {
    before(function () {
      console.log('\n%cPermute', styles.h2)
    })

    it('[core.Permute.0] should be able to go from shape [3, 2] -> [2, 3]', function () {
      const key = 'core.Permute.0'
      console.log(`\n%c[${key}] shape [3, 2] -> [2, 3]`, styles.h3)
      let testLayer = new layers.Permute([2, 1])
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

    it('[core.Permute.1] should be able to go from shape [2, 3, 4] -> [4, 3, 2]', function () {
      const key = 'core.Permute.1'
      console.log(`\n%c[${key}] shape [2, 3, 4] -> [4, 3, 2]`, styles.h3)
      let testLayer = new layers.Permute([3, 2, 1])
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
  * RepeatVector
  *********************************************************/

  describe('RepeatVector', function () {
    before(function () {
      console.log('\n%cRepeatVector', styles.h2)
    })

    it('[core.RepeatVector.0] should be able to go from shape [6] -> [7, 6]', function () {
      const key = 'core.RepeatVector.0'
      console.log(`\n%c[${key}] repeat vector, shape [6] -> [7, 6]`, styles.h3)
      let testLayer = new layers.RepeatVector(7)
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
  * Merge
  *********************************************************/

  describe('Merge', function () {
    before(function () {
      console.log('\n%cMerge', styles.h2)
    })

    it('[core.Merge.0] should produce expected values in sum mode', function () {
      const key = 'core.Merge.0'
      console.log(`\n%c[${key}] mode: sum`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'sum' })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
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

    it('[core.Merge.1] should produce expected values in mul mode', function () {
      const key = 'core.Merge.1'
      console.log(`\n%c[${key}] mode: mul`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'mul' })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
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

    it('[core.Merge.2] should produce expected values in ave mode', function () {
      const key = 'core.Merge.2'
      console.log(`\n%c[${key}] mode: ave`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'ave' })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
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

    it('[core.Merge.3] should produce expected values in max mode', function () {
      const key = 'core.Merge.3'
      console.log(`\n%c[${key}] mode: max`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'max' })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t1a = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let t1b = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
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

    it('[core.Merge.4] should produce expected values in concat mode (1D)', function () {
      const key = 'core.Merge.4'
      console.log(`\n%c[${key}] mode: concat (1D)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'concat', concatAxis: -1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      tb = testLayer1b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.5] should produce expected values in concat mode (2D, concatAxis=-1)', function () {
      const key = 'core.Merge.5'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=-1)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: -1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.6] should produce expected values in concat mode (2D, concatAxis=-2)', function () {
      const key = 'core.Merge.6'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=-2)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: -2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.7] should produce expected values in concat mode (2D, concatAxis=1)', function () {
      const key = 'core.Merge.7'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=1)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: 1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.8] should produce expected values in concat mode (2D, concatAxis=2)', function () {
      const key = 'core.Merge.8'
      console.log(`\n%c[${key}] mode: concat (2D, concatAxis=2)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'concat', concatAxis: 2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.9] should produce expected values in dot mode (2D x 2D, dotAxes=1)', function () {
      const key = 'core.Merge.9'
      console.log(`\n%c[${key}] mode: dot (2D x 2D, dotAxes=1)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'dot', dotAxes: 1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.10] should produce expected values in dot mode (2D x 2D, dotAxes=2)', function () {
      const key = 'core.Merge.10'
      console.log(`\n%c[${key}] mode: dot (2D x 2D, dotAxes=2)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'dot', dotAxes: 2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.11] should produce expected values in cos mode (2D x 2D, dotAxes=1)', function () {
      const key = 'core.Merge.11'
      console.log(`\n%c[${key}] mode: cos (2D x 2D, dotAxes=1)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'cos', dotAxes: 1 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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

    it('[core.Merge.12] should produce expected values in cos mode (2D x 2D, dotAxes=2)', function () {
      const key = 'core.Merge.12'
      console.log(`\n%c[${key}] mode: cos (2D x 2D, dotAxes=2)`, styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer2a = new layers.RepeatVector(3)
      let testLayer1b = new layers.Dense(2)
      let testLayer2b = new layers.RepeatVector(3)
      let testLayer3 = new layers.Merge({ mode: 'cos', dotAxes: 2 })
      testLayer1a.setWeights(TEST_DATA[key].weights.slice(0, 2).map(w => new KerasJS.Tensor(w.data, w.shape)))
      testLayer1b.setWeights(TEST_DATA[key].weights.slice(2, 4).map(w => new KerasJS.Tensor(w.data, w.shape)))
      let ta = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      let tb = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      ta = testLayer1a.call(ta)
      ta = testLayer2a.call(ta)
      tb = testLayer1b.call(tb)
      tb = testLayer2b.call(tb)
      console.log('%cin', styles.h4, stringifyCondensed([ta.tensor, tb.tensor]))
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
