/* eslint-env browser, mocha */

describe('recurrent layer: SimpleRNN', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [3, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: false, goBackwards: false, stateful: false }
    },
    {
      inputShape: [8, 5],
      attrs: { outputDim: 5, activation: 'sigmoid', returnSequences: false, goBackwards: false, stateful: false }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: true, goBackwards: false, stateful: false }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: false, goBackwards: true, stateful: false }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: true, goBackwards: true, stateful: false }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: false, goBackwards: false, stateful: true }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: true, goBackwards: false, stateful: true }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: false, goBackwards: true, stateful: true }
    },
    {
      inputShape: [7, 6],
      attrs: { outputDim: 4, activation: 'tanh', returnSequences: true, goBackwards: true, stateful: true }
    }
  ]

  before(function () {
    console.log('\n%crecurrent layer: SimpleRNN', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/

  describe('CPU', function () {
    before(function () {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `recurrent.SimpleRNN.${i}`
      const title = `[${key}] [CPU] test: ${inputShape[0]}x${inputShape[1]} input, activation='${attrs.activation}', returnSequences=${attrs.returnSequences}, goBackwards=${attrs.goBackwards}, stateful=${attrs.stateful}`

      it(title, function () {
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

  describe('GPU', function () {
    before(function () {
      console.log('\n%cGPU', styles.h2)
    })

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `recurrent.SimpleRNN.${i}`
      const title = `[${key}] [GPU] test: ${inputShape[0]}x${inputShape[1]} input, activation='${attrs.activation}', returnSequences=${attrs.returnSequences}, goBackwards=${attrs.goBackwards}, stateful=${attrs.stateful}`

      it(title, function () {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.SimpleRNN(attrs)
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape, { gpu: true })
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        const startTime = performance.now()

        // To test statefulness, we run call() twice (see corresponding jupyter notebook)
        t = testLayer.call(t)
        if (attrs.stateful) {
          t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape, { gpu: true })
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
