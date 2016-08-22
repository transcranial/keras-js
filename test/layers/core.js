/* eslint-env browser, mocha */

describe('Layers: Core', function () {
  const assert = chai.assert
  const styles = testUtils.styles
  const approxEquals = testUtils.approxEquals
  const logTime = testUtils.logTime

  const layers = KerasJS.layers

  /*********************************************************
  * Dense
  *********************************************************/

  describe('Dense', function () {
    it('should produce expected values', function () {
      console.log('\n%Layers: Core', styles.h1)
      console.log('\n%cDense', styles.h2)
      console.log('\n%ctest 1', styles.h3)
      let testLayer = new layers.Dense(2)
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([7.3, -0.21])
      const shapeExpected = [2]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should produce expected values, with sigmoid activation function', function () {
      console.log('\n%ctest 2 (with sigmoid activation)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'sigmoid' })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2]),
        new KerasJS.Tensor([0.5, 0.7], [2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.999325, 0.447692])
      const shapeExpected = [2]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should produce expected values, with softplus activation function and no bias', function () {
      console.log('\n%ctest 3 (with softplus activation and no bias)', styles.h3)
      let testLayer = new layers.Dense(2, { activation: 'softplus', bias: false })
      testLayer.setWeights([
        new KerasJS.Tensor([0.1, 0.4, 0.5, 0.1, 1, -2, 0, 0.3, 0.2, 0.1, 3, 0], [6, 2])
      ])
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([6.801113, 0.338274])
      const shapeExpected = [2]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })
})
