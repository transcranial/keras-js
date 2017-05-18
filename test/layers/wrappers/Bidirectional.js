describe('wrappers layer: Bidirectional', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      wrappedLayer: 'SimpleRNN',
      inputShape: [3, 6],
      attrs: { merge_mode: 'sum' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', return_sequences: false }
    },
    {
      wrappedLayer: 'SimpleRNN',
      inputShape: [3, 6],
      attrs: { merge_mode: 'mul' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', return_sequences: false }
    },
    {
      wrappedLayer: 'SimpleRNN',
      inputShape: [3, 6],
      attrs: { merge_mode: 'concat' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', return_sequences: false }
    },
    {
      wrappedLayer: 'SimpleRNN',
      inputShape: [3, 6],
      attrs: { merge_mode: 'ave' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', return_sequences: false }
    },
    {
      wrappedLayer: 'SimpleRNN',
      inputShape: [3, 6],
      attrs: { merge_mode: 'concat' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', return_sequences: true }
    },
    {
      wrappedLayer: 'GRU',
      inputShape: [3, 6],
      attrs: { merge_mode: 'concat' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', recurrent_activation: 'hard_sigmoid', return_sequences: true }
    },
    {
      wrappedLayer: 'LSTM',
      inputShape: [3, 6],
      attrs: { merge_mode: 'concat' },
      wrappedLayerAttrs: { units: 4, activation: 'tanh', recurrent_activation: 'hard_sigmoid', return_sequences: true }
    }
  ]

  before(function() {
    console.log('\n%cwrappers layer: Bidirectional', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ wrappedLayer, inputShape, attrs, wrappedLayerAttrs }, i) => {
      const key = `wrappers.Bidirectional.${i}`
      const title = `[${key}] [CPU] test: ${inputShape} input, merge_mode: ${attrs.merge_mode}, wrapped layer: ${wrappedLayer}, wrapped layer attrs: ${JSON.stringify(wrappedLayerAttrs)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.Bidirectional(
          Object.assign(attrs, { layer: new layers[wrappedLayer](wrappedLayerAttrs) })
        )
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

    testParams.forEach(({ wrappedLayer, inputShape, attrs, wrappedLayerAttrs }, i) => {
      const key = `wrappers.Bidirectional.${i}`
      const title = `[${key}] [GPU] test: ${inputShape} input, merge_mode: ${attrs.merge_mode}, wrapped layer: ${wrappedLayer}, wrapped layer attrs: ${JSON.stringify(wrappedLayerAttrs)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.Bidirectional(
          Object.assign(attrs, { layer: new layers[wrappedLayer](wrappedLayerAttrs), gpu: true })
        )
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
