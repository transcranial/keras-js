describe('core layer: Activation', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const activations = [
    'softmax',
    'elu',
    'selu',
    'softplus',
    'softsign',
    'relu',
    'tanh',
    'sigmoid',
    'hard_sigmoid',
    'linear'
  ]

  before(function() {
    console.log('\n%ccore layer: Activation', styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    activations.forEach((activation, i) => {
      const key = `core.Activation.${i}`
      const title = `[${key}] [CPU] should produce expected values for ${activation} activation following Dense layer`

      it(title, function() {
        console.log(`\n%c[${key}] [CPU] ${activation}`, styles.h3)
        let testLayer1 = new layers.Dense({ units: 2 })
        testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
        t = testLayer1.call(t)
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        let testLayer2 = new layers.Activation({ activation })
        const startTime = performance.now()
        t = testLayer2.call(t)
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

    activations.forEach((activation, i) => {
      const key = `core.Activation.${i}`
      const title = `[${key}] [CPU] should produce expected values for ${activation} activation following Dense layer`

      it(title, function() {
        console.log(`\n%c[${key}] [GPU] ${activation}`, styles.h3)
        let testLayer1 = new layers.Dense({ units: 2, gpu: true })
        testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
        t = testLayer1.call(t)
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
        let testLayer2 = new layers.Activation({ activation, gpu: true })
        const startTime = performance.now()
        t = testLayer2.call(t)
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
