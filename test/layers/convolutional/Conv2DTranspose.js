describe('convolutional layer: Conv2DTranspose', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 4,
        kernel_size: [3, 3],
        strides: [1, 1],
        padding: 'valid',
        data_format: 'channels_last',
        activation: 'linear',
        use_bias: false
      }
    },
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 4,
        kernel_size: [3, 3],
        strides: [1, 1],
        padding: 'valid',
        data_format: 'channels_last',
        activation: 'linear',
        use_bias: true
      }
    },
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 4,
        kernel_size: [3, 3],
        strides: [2, 2],
        padding: 'valid',
        data_format: 'channels_last',
        activation: 'relu',
        use_bias: true
      }
    },
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 4,
        kernel_size: [3, 3],
        strides: [1, 1],
        padding: 'same',
        data_format: 'channels_last',
        activation: 'relu',
        use_bias: true
      }
    },
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 5,
        kernel_size: [3, 3],
        strides: [2, 2],
        padding: 'same',
        data_format: 'channels_last',
        activation: 'relu',
        use_bias: true
      }
    },
    {
      inputShape: [4, 4, 2],
      attrs: {
        filters: 3,
        kernel_size: [2, 3],
        strides: [1, 1],
        padding: 'same',
        data_format: 'channels_last',
        activation: 'relu',
        use_bias: true
      }
    }
  ]

  before(function() {
    console.log('\n%cconvolutional layer: Conv2DTranspose', styles.h1)
  })

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `convolutional.Conv2DTranspose.${i}`
      const title = `[${key}] [CPU] test: ${attrs.filters} ${attrs.kernel_size} filters on ${inputShape} input, strides=${attrs.strides}, padding='${attrs.padding}', data_format='${attrs.data_format}', activation='${attrs.activation}', use_bias=${attrs.use_bias}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.Conv2DTranspose(attrs)
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
      const key = `convolutional.Conv2DTranspose.${i}`
      const title = `[${key}] [GPU] test: ${attrs.filters} ${attrs.kernel_size} filters on ${inputShape} input, strides=${attrs.strides}, padding='${attrs.padding}', data_format='${attrs.data_format}', activation='${attrs.activation}', use_bias=${attrs.use_bias}`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.Conv2DTranspose(Object.assign(attrs, { gpu: true }))
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
