/* eslint-env browser, mocha */

describe('Layers: Advanced Activations', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function () {
    console.log('\n%Layers: Advanced Activations', styles.h1)
  })

  /*********************************************************
  * LeakyReLU
  *********************************************************/

  describe('LeakyReLU', function () {
    before(function () {
      console.log('\n%cLeakyReLU', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%calpha=0.4', styles.h3)
      let testLayer = new layers.LeakyReLU(0.4)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, -0.2, -0.04, 1.0, 2.0])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
  * PReLU
  *********************************************************/

  describe('PReLU', function () {
    before(function () {
      console.log('\n%cPReLU', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%cweights: alphas', styles.h3)
      let testLayer = new layers.PReLU()
      testLayer.setWeights([
        new KerasJS.Tensor([-0.03, -0.02, 0.02, -0.03, -0.03, -0.01], [6])
      ])
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, -0.01, 0.003, 1.0, 2.0])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
  * ELU
  *********************************************************/

  describe('ELU', function () {
    before(function () {
      console.log('\n%cELU', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%calpha=1.1', styles.h3)
      let testLayer = new layers.ELU(1.1)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.2, -0.432816, -0.104679, 1.0, 2.0])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
  * ParametricSoftplus
  *********************************************************/

  describe('ParametricSoftplus', function () {
    before(function () {
      console.log('\n%cParametricSoftplus', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%cweights: alphas, betas', styles.h3)
      let testLayer = new layers.ParametricSoftplus()
      testLayer.setWeights([
        new KerasJS.Tensor([0.13, -0.02, 0.02, -0.03, -0.03, -0.01], [6]),
        new KerasJS.Tensor([-0.03, -0.1, 0.02, 0.5, 0.2, 0.0], [6])
      ])
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.090109, -0.013664, 0.013763, -0.020054, -0.023944, -0.006931])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
  * ThresholdedReLU
  *********************************************************/

  describe('ThresholdedReLU', function () {
    before(function () {
      console.log('\n%cThresholdedReLU', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%theta=0.9', styles.h3)
      let testLayer = new layers.ThresholdedReLU(0.9)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
  * SReLU
  *********************************************************/

  describe('SReLU', function () {
    before(function () {
      console.log('\n%cSReLU', styles.h2)
    })

    it('should produce expected values', function () {
      console.log('\n%cweights: t_left, a_left, t_right, a_right', styles.h3)
      let testLayer = new layers.SReLU()
      testLayer.setWeights([
        new KerasJS.Tensor([0.13, -0.02, 0.02, -0.03, -0.03, -0.01], [6]),
        new KerasJS.Tensor([-0.03, -0.1, 0.02, 0.5, 0.2, 0.0], [6]),
        new KerasJS.Tensor([-0.9, 0.8, 0.0, -1.0, 0.7, 0.4], [6]),
        new KerasJS.Tensor([0.1, 0.2, 0.3, 0.0, 0.5, -0.2], [6])
      ])
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array([0.1339, 0.2, 0.0096, -0.065, 0.835, 0.068])
      const shapeExpected = [6]
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })
})
