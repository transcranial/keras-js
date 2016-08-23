/* eslint-env browser, mocha */

describe('Layers: Core', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function () {
    console.log('\n%Layers: Core', styles.h1)
  })

  /*********************************************************
  * Dense
  *********************************************************/

  describe('Dense', function () {
    before(function () {
      console.log('\n%cDense', styles.h2)
    })

    it('[CPU] should produce expected values', function () {
      console.log('\n%c[CPU] test 1', styles.h3)
      let testLayer = new layers.Dense(2)
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([7.3, -0.21])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[CPU] should produce expected values, with sigmoid activation function', function () {
      console.log('\n%c[CPU] test 2 (with sigmoid activation)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'sigmoid' })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.999325, 0.447692])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[CPU] should produce expected values, with softplus activation function and no bias', function () {
      console.log('\n%c[CPU] test 3 (with softplus activation and no bias)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'softplus', bias: false })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([6.801113, 0.338274])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[GPU] should produce expected values', function () {
      console.log('\n%c[GPU] test 1', styles.h3)
      let testLayer = new layers.Dense(2)
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6], { useWeblas: true })
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([7.3, -0.21])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[GPU] should produce expected values, with sigmoid activation function', function () {
      console.log('\n%c[GPU] test 2 (with sigmoid activation)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'sigmoid' })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6], { useWeblas: true })
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.999325, 0.447692])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('[GPU] should produce expected values, with softplus activation function and no bias', function () {
      console.log('\n%c[GPU] test 3 (with softplus activation and no bias)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'softplus', bias: false })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6], { useWeblas: true })
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([6.801113, 0.338274])
      const shapeExpected = [2]
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

    it('should produce expected values for tanh activation following Dense layer', function () {
      console.log('\n%ctest 1 (tanh)', styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation('tanh')
      const startTime = performance.now()
      t = testLayer2.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.999999, -0.206966])
      const shapeExpected = [2]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should produce expected values for hardSigmoid activation following Dense layer', function () {
      console.log('\n%ctest 2 (hardSigmoid)', styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Activation('hardSigmoid')
      const startTime = performance.now()
      t = testLayer2.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([1.0, 0.458])
      const shapeExpected = [2]
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

    it('should just pass through tensor during test time', function () {
      console.log('\n%cshould pass through', styles.h3)
      let testLayer1 = new layers.Dense(2)
      testLayer1.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t = testLayer1.call(t)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      let testLayer2 = new layers.Dropout(0.5)
      const startTime = performance.now()
      t = testLayer2.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([7.3, -0.21])
      const shapeExpected = [2]
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

    it('should do nothing for 1D', function () {
      console.log('\n%c1D', styles.h3)
      let testLayer = new layers.Flatten()
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should flatten 2D', function () {
      console.log('\n%c2D', styles.h3)
      let testLayer = new layers.Flatten()
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [3, 2])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should flatten 3D', function () {
      console.log('\n%c3D', styles.h3)
      let testLayer = new layers.Flatten()
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2], [3, 2, 2])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0])
      const shapeExpected = [12]
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

    it('should be able to go from shape [6] -> [2, 3]', function () {
      console.log('\n%cshape [6] -> [2, 3]', styles.h3)
      let testLayer = new layers.Reshape([2, 3])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2])
      const shapeExpected = [2, 3]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should be able to go from shape [3, 2] -> [6]', function () {
      console.log('\n%cshape [3, 2] -> [6]', styles.h3)
      let testLayer = new layers.Reshape([6])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [3, 2])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should be able to go from shape [3, 2, 2] -> [4, 3]', function () {
      console.log('\n%cshape [3, 2, 2] -> [4, 3]', styles.h3)
      let testLayer = new layers.Reshape([4, 3])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2], [3, 2, 2])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0])
      const shapeExpected = [4, 3]
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

    it('should be able to go from shape [3, 2] -> [2, 3]', function () {
      console.log('\n%cshape [3, 2] -> [2, 3]', styles.h3)
      let testLayer = new layers.Permute([2, 1])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [3, 2])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.5, 1.0, 0.2, -0.1, 2.0])
      const shapeExpected = [2, 3]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })

    it('should be able to go from shape [2, 3, 4] -> [4, 3, 2]', function () {
      console.log('\n%cshape [2, 3, 4] -> [4, 3, 2]', styles.h3)
      let testLayer = new layers.Permute([3, 2, 1])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2], [2, 3, 4])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.2, 0.2, 2.0, 2.0, -0.1, -0.1, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, -0.1, -0.1, 0.2, 0.2, 2.0, 2.0])
      const shapeExpected = [4, 3, 2]
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

    it('should be able to go from shape [6] -> [7, 6]', function () {
      console.log('\n%crepeat vector, shape [6] -> [7, 6]', styles.h3)
      let testLayer = new layers.RepeatVector(7)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0])
      const shapeExpected = [7, 6]
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

    it('should produce expected values in sum mode', function () {
      console.log('\n%cmode: sum', styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'sum' })
      testLayer1a.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      testLayer1b.setWeights([
        new KerasJS.Tensor([1, 0, -0.9, 0.6, -0.7, 0, 0.2, 0.4, 0, 0, -1, 2.3], [6, 2]),
        new KerasJS.Tensor([0.1, -0.2], [2])
      ])
      let t1a = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      let t1b = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([4.85, 4.27])
      const shapeExpected = [2]
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })

    it('should produce expected values in mul mode', function () {
      console.log('\n%cmode: mul', styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'mul' })
      testLayer1a.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      testLayer1b.setWeights([
        new KerasJS.Tensor([1, 0, -0.9, 0.6, -0.7, 0, 0.2, 0.4, 0, 0, -1, 2.3], [6, 2]),
        new KerasJS.Tensor([0.1, -0.2], [2])
      ])
      let t1a = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      let t1b = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([-17.885, -0.9408])
      const shapeExpected = [2]
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })

    it('should produce expected values in ave mode', function () {
      console.log('\n%cmode: ave', styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'ave' })
      testLayer1a.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      testLayer1b.setWeights([
        new KerasJS.Tensor([1, 0, -0.9, 0.6, -0.7, 0, 0.2, 0.4, 0, 0, -1, 2.3], [6, 2]),
        new KerasJS.Tensor([0.1, -0.2], [2])
      ])
      let t1a = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      let t1b = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([2.425, 2.135])
      const shapeExpected = [2]
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })

    it('should produce expected values in max mode', function () {
      console.log('\n%cmode: max', styles.h3)
      let testLayer1a = new layers.Dense(2)
      let testLayer1b = new layers.Dense(2)
      let testLayer2 = new layers.Merge({ mode: 'max' })
      testLayer1a.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      testLayer1b.setWeights([
        new KerasJS.Tensor([1, 0, -0.9, 0.6, -0.7, 0, 0.2, 0.4, 0, 0, -1, 2.3], [6, 2]),
        new KerasJS.Tensor([0.1, -0.2], [2])
      ])
      let t1a = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      let t1b = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      t1a = testLayer1a.call(t1a)
      t1b = testLayer1b.call(t1b)
      console.log('%cin', styles.h4, stringifyCondensed([t1a.tensor, t1b.tensor]))
      const startTime = performance.now()
      let t2 = testLayer2.call([t1a, t1b])
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t2.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([7.3, 4.48])
      const shapeExpected = [2]
      assert.deepEqual(t2.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t2.tensor, dataExpected))
    })
  })
})
