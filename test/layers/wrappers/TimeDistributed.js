describe('wrappers layer: TimeDistributed', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      wrappedLayer: 'Dense',
      inputShape: [3, 6],
      wrappedLayerAttrs: { units: 4, activation: 'linear', input_dim: null, use_bias: true }
    },
    {
      wrappedLayer: 'Conv2D',
      inputShape: [5, 4, 4, 2],
      wrappedLayerAttrs: {
        filters: 6,
        kernel_size: [3, 3],
        strides: [1, 1],
        padding: 'valid',
        data_format: 'channels_last',
        dilation_rate: [1, 1],
        activation: 'linear',
        use_bias: true
      }
    }
  ]

  before(function() {
    console.log('\n%cwrappers layer: TimeDistributed', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ wrappedLayer, inputShape, wrappedLayerAttrs }, i) => {
      const key = `wrappers.TimeDistributed.${i}`
      const title = `[${key}] [CPU] test: ${inputShape} input, wrapped layer: ${wrappedLayer}, wrapped layer attrs: ${JSON.stringify(wrappedLayerAttrs)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.TimeDistributed({ layer: new layers[wrappedLayer](wrappedLayerAttrs) })
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
    })
  })

  /*********************************************************
  * GPU
  *********************************************************/
  describe('GPU', function() {
    before(function() {
      console.log('\n%cGPU', styles.h2)
    })

    testParams.forEach(({ wrappedLayer, inputShape, wrappedLayerAttrs }, i) => {
      const key = `wrappers.TimeDistributed.${i}`
      const title = `[${key}] [GPU] test: ${inputShape} input, wrapped layer: ${wrappedLayer}, wrapped layer attrs: ${JSON.stringify(wrappedLayerAttrs)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.TimeDistributed({ layer: new layers[wrappedLayer](wrappedLayerAttrs), gpu: true })
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
    })
  })
})
