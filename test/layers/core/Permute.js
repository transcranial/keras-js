describe('core layer: Permute', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [3, 2],
      expectedOutputShape: [2, 3],
      attrs: {
        dims: [2, 1]
      }
    },
    {
      inputShape: [2, 3, 4],
      expectedOutputShape: [4, 3, 2],
      attrs: {
        dims: [3, 2, 1]
      }
    },
    {
      inputShape: [1, 6, 4],
      expectedOutputShape: [6, 1, 4],
      attrs: {
        dims: [2, 1, 3]
      }
    },
    {
      inputShape: [1, 3, 4, 2],
      expectedOutputShape: [4, 1, 3, 2],
      attrs: {
        dims: [3, 1, 2, 4]
      }
    }
  ]

  before(function() {
    console.log('\n%ccore layer: Permute', styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ inputShape, expectedOutputShape, attrs }, i) => {
      const key = `core.Permute.${i}`
      const title = `[${key}] [CPU] should be able to go from shape [${inputShape}] -> [${expectedOutputShape}]`

      it(title, function() {
        console.log(`\n%c[${key}] [CPU] shape [${inputShape}] -> [${expectedOutputShape}]`, styles.h3)
        let testLayer = new layers.Permute(attrs)
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
  })

  /*********************************************************
   * GPU
   *********************************************************/
  describe('GPU', function() {
    before(function() {
      console.log('\n%cGPU', styles.h2)
    })

    testParams.forEach(({ inputShape, expectedOutputShape, attrs }, i) => {
      const key = `core.Permute.${i}`
      const title = `[${key}] [GPU] should be able to go from shape [${inputShape}] -> [${expectedOutputShape}]`

      it(title, function() {
        console.log(`\n%c[${key}] [CPU] shape [${inputShape}] -> [${expectedOutputShape}]`, styles.h3)
        let testLayer = new layers.Permute(Object.assign(attrs, { gpu: true }))
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
  })
})
