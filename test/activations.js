/* eslint-env browser, mocha */

const assert = chai.assert
const activations = KerasJS.activations

const styles = testUtils.styles
const approxEquals = testUtils.approxEquals
const logTime = testUtils.logTime

describe('activations', function () {
  describe('softmax', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%cactivations', styles.h1)
      console.log('\n%csoftmax', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softmax(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.067194, 0.082071, 0.110784, 0.0608, 0.182652, 0.4965])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softmax(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.067194, 0.082071, 0.110784, 0.0608, 0.182652, 0.4965, 0.107768, 0.149902, 0.11105, 0.247147, 0.082268, 0.301865])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  describe('relu', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%crelu', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.relu(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.2, 0.5, 0.0, 1.0, 2.0])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.relu(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.2, 0.5, 0.0, 1.0, 2.0, 0.0, 0.3, 0.0, 0.8, 0.0, 1.0])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.relu(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.2, 0.0, 0.0, 1.0, 2.0, 0.0, 2.3, 0.0, 0.8, 0.0, 1.0])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work with maxValue', function () {
      console.log('\n%c3D, maxValue=0.5', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.relu(t, { maxValue: 0.5 })
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.2, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work with alpha (slope of negative portion)', function () {
      console.log('\n%c3D, alpha=0.3', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.relu(t, { alpha: 0.3 })
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.2, -0.15, -0.03, 1.0, 2.0, -0.009, 2.3, 0.0, 0.8, -0.09, 1.0])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })
})
