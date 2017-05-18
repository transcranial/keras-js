describe('recurrent layer: SimpleRNN', function() {
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
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: false,
        go_backwards: false,
        stateful: false
      }
    },
    {
      inputShape: [8, 5],
      attrs: {
        units: 5,
        activation: 'sigmoid',
        use_bias: true,
        return_sequences: false,
        go_backwards: false,
        stateful: false
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: true,
        go_backwards: false,
        stateful: false
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: false,
        go_backwards: true,
        stateful: false
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: true,
        go_backwards: true,
        stateful: false
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: false,
        go_backwards: false,
        stateful: true
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: true,
        go_backwards: false,
        stateful: true
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: true,
        return_sequences: false,
        go_backwards: true,
        stateful: true
      }
    },
    {
      inputShape: [7, 6],
      attrs: {
        units: 4,
        activation: 'tanh',
        use_bias: false,
        return_sequences: true,
        go_backwards: true,
        stateful: true
      }
    }
  ]

  before(function() {
    console.log('\n%crecurrent layer: SimpleRNN', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `recurrent.SimpleRNN.${i}`
      const title = `[${key}] [CPU] test: ${inputShape} input, activation='${attrs.activation}', use_bias=${attrs.use_bias}, return_sequences=${attrs.return_sequences}, go_backwards=${attrs.go_backwards}, stateful=${attrs.stateful}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.SimpleRNN(attrs)
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        const startTime = performance.now()

        // To test statefulness, we run call() twice (see corresponding jupyter notebook)
        t = testLayer.call(t)
        if (attrs.stateful) {
          t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
          t = testLayer.call(t)
        }

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
      const key = `recurrent.SimpleRNN.${i}`
      const title = `[${key}] [GPU] test: ${inputShape} input, activation='${attrs.activation}', use_bias=${attrs.use_bias}, return_sequences=${attrs.return_sequences}, go_backwards=${attrs.go_backwards}, stateful=${attrs.stateful}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.SimpleRNN(Object.assign(attrs, { gpu: true }))
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        const startTime = performance.now()

        // To test statefulness, we run call() twice (see corresponding jupyter notebook)
        t = testLayer.call(t)
        if (attrs.stateful) {
          t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
          t = testLayer.call(t)
        }

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
