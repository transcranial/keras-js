/* eslint-env browser, mocha */

const assert = chai.assert
const activations = KerasJS.activations

const styles = testUtils.styles
const approxEquals = testUtils.approxEquals
const logTime = testUtils.logTime

describe('activations', function () {
  /*********************************************************
   * softmax
   *********************************************************/

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

  /*********************************************************
   * softplus
   *********************************************************/

  describe('softplus', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%csoftplus', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softplus(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.693147, 0.798139, 0.974077, 0.644397, 1.313262, 2.126928])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softplus(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.693147, 0.798139, 0.974077, 0.644397, 1.313262, 2.126928, 0.67826, 0.854355, 0.693147, 1.171101, 0.554355, 1.313262])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softplus(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.693147, 0.798139, 0.474077, 0.644397, 1.313262, 2.126928, 0.67826, 2.395545, 0.693147, 1.171101, 0.554355, 1.313262])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  /*********************************************************
   * softsign
   *********************************************************/

  describe('softsign', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%csoftsign', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softsign(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.166667, 0.333333, -0.090909, 0.5, 0.666667])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softsign(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.166667, 0.333333, -0.090909, 0.5, 0.666667, -0.029126, 0.230769, 0.0, 0.444444, -0.230769, 0.5])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.softsign(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.166667, -0.333333, -0.090909, 0.5, 0.666667, -0.029126, 0.69697, 0.0, 0.444444, -0.230769, 0.5])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  /*********************************************************
   * relu
   *********************************************************/

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

  /*********************************************************
   * tanh
   *********************************************************/

  describe('tanh', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%ctanh', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.tanh(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.197375, 0.462117, -0.099668, 0.761594, 0.964028])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.tanh(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.197375, 0.462117, -0.099668, 0.761594, 0.964028, -0.029991, 0.291313, 0.0, 0.664037, -0.291313, 0.761594])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.tanh(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.0, 0.197375, -0.462117, -0.099668, 0.761594, 0.964028, -0.029991, 0.980096, 0.0, 0.664037, -0.291313, 0.761594])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  /*********************************************************
   * sigmoid
   *********************************************************/

  describe('sigmoid', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%csigmoid', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.sigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.549834, 0.622459, 0.475021, 0.731059, 0.880797])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.sigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.549834, 0.622459, 0.475021, 0.731059, 0.880797, 0.492501, 0.574443, 0.5, 0.689974, 0.425557, 0.731059])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.sigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.549834, 0.377541, 0.475021, 0.731059, 0.880797, 0.492501, 0.908877, 0.5, 0.689974, 0.425557, 0.731059])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  /*********************************************************
   * hardSigmoid
   *********************************************************/

  describe('hardSigmoid', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%chardSigmoid', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.hardSigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.54, 0.6, 0.48, 0.7, 0.9])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.hardSigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.54, 0.6, 0.48, 0.7, 0.9, 0.494, 0.56, 0.5, 0.66, 0.44, 0.7])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.hardSigmoid(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0.5, 0.54, 0.4, 0.48, 0.7, 0.9, 0.494, 0.96, 0.5, 0.66, 0.44, 0.7])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })

  /*********************************************************
   * linear
   *********************************************************/

  describe('linear', function () {
    it('should work for 1D tensor', function () {
      console.log('\n%clinear', styles.h2)
      console.log('\n%c1D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2], [6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.linear(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2])
      const shapeExpected = [6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 2D tensor', function () {
      console.log('\n%c2D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1], [2, 6])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.linear(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0, 0.2, 0.5, -0.1, 1, 2, -0.03, 0.3, 0, 0.8, -0.3, 1])
      const shapeExpected = [2, 6]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })

    it('should work for 3D tensor', function () {
      console.log('\n%c3D', styles.h3)
      let t = new KerasJS.Tensor([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1], [2, 2, 3])
      console.log('%cin', styles.h4, t)
      const startTime = performance.now()
      activations.linear(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, t)
      logTime(startTime, endTime)
      const dataOut = t.tensor.data
      const shapeOut = t.tensor.shape
      const dataExpected = new Float32Array([0, 0.2, -0.5, -0.1, 1, 2, -0.03, 2.3, 0, 0.8, -0.3, 1])
      const shapeExpected = [2, 2, 3]
      assert.deepEqual(shapeOut, shapeExpected)
      assert.isTrue(approxEquals(dataOut, dataExpected))
    })
  })
})
