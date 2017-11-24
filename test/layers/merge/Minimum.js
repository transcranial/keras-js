describe('merge layer: Minimum', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { numBranches: 2, attrs: { units: 2 } },
    { numBranches: 3, attrs: { units: 2 } },
    { numBranches: 4, attrs: { units: 2 } }
  ]

  before(function() {
    console.log('\n%cmerge layer: Minimum', styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ numBranches, attrs }, i) => {
      const key = `merge.Minimum.${i}`
      const title = `[${key}] [CPU] num branches merging: ${numBranches}`
      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        const testInputs = []
        for (let i = 0; i < numBranches; i++) {
          const layer = new layers.Dense(attrs)
          layer.setWeights(
            TEST_DATA[key].weights.slice(2 * i, 2 * (i + 1)).map(w => new KerasJS.Tensor(w.data, w.shape))
          )
          let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
          t = layer.call(t)
          testInputs.push(t)
          console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        }
        const mergeLayer = new layers.Minimum()
        const startTime = performance.now()
        const output = mergeLayer.call(testInputs)
        const endTime = performance.now()
        console.log('%cout', styles.h4, stringifyCondensed(output.tensor))
        logTime(startTime, endTime)
        const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
        const shapeExpected = TEST_DATA[key].expected.shape
        assert.deepEqual(output.tensor.shape, shapeExpected)
        assert.isTrue(approxEquals(output.tensor, dataExpected))
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

    testParams.forEach(({ numBranches, attrs }, i) => {
      const key = `merge.Minimum.${i}`
      const title = `[${key}] [GPU] num branches merging: ${numBranches}`
      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        const testInputs = []
        for (let i = 0; i < numBranches; i++) {
          const layer = new layers.Dense(Object.assign(attrs, { gpu: true }))
          layer.setWeights(
            TEST_DATA[key].weights.slice(2 * i, 2 * (i + 1)).map(w => new KerasJS.Tensor(w.data, w.shape))
          )
          let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
          t = layer.call(t)
          testInputs.push(t)
          console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        }
        const mergeLayer = new layers.Minimum({ gpu: true })
        const startTime = performance.now()
        const output = mergeLayer.call(testInputs)
        const endTime = performance.now()
        console.log('%cout', styles.h4, stringifyCondensed(output.tensor))
        logTime(startTime, endTime)
        const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
        const shapeExpected = TEST_DATA[key].expected.shape
        assert.deepEqual(output.tensor.shape, shapeExpected)
        assert.isTrue(approxEquals(output.tensor, dataExpected))
      })
    })
  })
})
