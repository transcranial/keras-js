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
})
