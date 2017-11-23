describe('wrappers layer: Bidirectional', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: false } },
        merge_mode: 'sum'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: false } },
        merge_mode: 'mul'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: false } },
        merge_mode: 'concat'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: false } },
        merge_mode: 'ave'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: true } },
        merge_mode: 'concat'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: {
          class_name: 'GRU',
          config: { units: 4, activation: 'tanh', recurrent_activation: 'hard_sigmoid', return_sequences: true }
        },
        merge_mode: 'concat'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: {
          class_name: 'LSTM',
          config: { units: 4, activation: 'tanh', recurrent_activation: 'hard_sigmoid', return_sequences: true }
        },
        merge_mode: 'concat'
      }
    },
    {
      inputShape: [3, 6],
      attrs: {
        layer: { class_name: 'SimpleRNN', config: { units: 4, activation: 'tanh', return_sequences: true } },
        merge_mode: 'sum'
      }
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

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `wrappers.Bidirectional.${i}`
      const title =
        `[${key}] [CPU] test: ${JSON.stringify(inputShape)} input, ` +
        `merge_mode: ${attrs.merge_mode}, wrapped layer: ${attrs.layer.class_name}, ` +
        `wrapped layer attrs: ${JSON.stringify(attrs.layer.config)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        const testLayer = new layers.Bidirectional(attrs)
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

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `wrappers.Bidirectional.${i}`
      const title =
        `[${key}] [GPU] test: ${JSON.stringify(inputShape)} input, ` +
        `merge_mode: ${attrs.merge_mode}, wrapped layer: ${attrs.layer.class_name}, ` +
        `wrapped layer attrs: ${JSON.stringify(attrs.layer.config)}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        const testLayer = new layers.Bidirectional(Object.assign(attrs, { gpu: true }))
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
