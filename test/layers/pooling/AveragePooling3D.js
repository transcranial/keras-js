describe('pooling layer: AveragePooling3D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [2, 2, 2], strides: null, padding: 'valid', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [2, 2, 2], strides: [1, 1, 1], padding: 'valid', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 5, 2, 3],
      attrs: { pool_size: [2, 2, 2], strides: [2, 1, 1], padding: 'valid', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [3, 3, 3], strides: null, padding: 'valid', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [3, 3, 3], strides: [3, 3, 3], padding: 'valid', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [2, 2, 2], strides: null, padding: 'same', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [2, 2, 2], strides: [1, 1, 1], padding: 'same', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 5, 4, 2],
      attrs: { pool_size: [2, 2, 2], strides: [1, 2, 1], padding: 'same', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [3, 3, 3], strides: null, padding: 'same', data_format: 'channels_last' }
    },
    {
      inputShape: [4, 4, 4, 2],
      attrs: { pool_size: [3, 3, 3], strides: [3, 3, 3], padding: 'same', data_format: 'channels_last' }
    },
    {
      inputShape: [2, 3, 3, 4],
      attrs: { pool_size: [3, 3, 3], strides: [2, 2, 2], padding: 'valid', data_format: 'channels_first' }
    },
    {
      inputShape: [2, 3, 3, 4],
      attrs: { pool_size: [3, 3, 3], strides: [1, 1, 1], padding: 'same', data_format: 'channels_first' }
    },
    {
      inputShape: [3, 4, 4, 3],
      attrs: { pool_size: [2, 2, 2], strides: null, padding: 'valid', data_format: 'channels_first' }
    }
  ]

  before(function() {
    console.log('\n%cpooling layer: AveragePooling3D', styles.h1)
  })

  testParams.forEach(({ inputShape, attrs }, i) => {
    const key = `pooling.AveragePooling3D.${i}`
    const title = `[${key}] test: ${inputShape} input, pool_size='${attrs.pool_size}', strides=${attrs.strides}, padding=${attrs.padding}, dimOrdering=${attrs.dimOrdering}`

    it(title, function() {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.AveragePooling3D(attrs)
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
