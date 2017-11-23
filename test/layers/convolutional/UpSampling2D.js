describe('convolutional layer: UpSampling2D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { inputShape: [3, 3, 3], attrs: { size: [2, 2], data_format: 'channels_last' } },
    { inputShape: [3, 3, 3], attrs: { size: [2, 2], data_format: 'channels_first' } },
    { inputShape: [4, 2, 2], attrs: { size: [3, 2], data_format: 'channels_last' } },
    { inputShape: [4, 3, 2], attrs: { size: [1, 3], data_format: 'channels_first' } },
    { inputShape: [1, 3, 2], attrs: { size: 2, data_format: 'channels_last' } }
  ]

  before(function() {
    console.log('\n%cconvolutional layer: UpSampling2D', styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2)
    })

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `convolutional.UpSampling2D.${i}`
      const title = `[${key}] [CPU] size ${attrs.size} upsampling on ${JSON.stringify(
        inputShape
      )} input, data_format='${attrs.data_format}'`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.UpSampling2D(attrs)
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
      const key = `convolutional.UpSampling2D.${i}`
      const title = `[${key}] [GPU] size ${attrs.size} upsampling on ${JSON.stringify(
        inputShape
      )} input, data_format='${attrs.data_format}'`

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3)
        let testLayer = new layers.UpSampling2D(Object.assign(attrs, { gpu: true }))
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
