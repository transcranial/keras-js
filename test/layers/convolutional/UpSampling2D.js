describe('convolutional layer: UpSampling2D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cconvolutional layer: UpSampling2D', styles.h1)
  })

  it(`[convolutional.UpSampling2D.0] size 2x2 upsampling on 3x3x3 input, data_format='channels_last'`, function() {
    const key = `convolutional.UpSampling2D.0`
    console.log(`\n%c[${key}] size 2x2 upsampling on 3x3x3 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.UpSampling2D({ size: [2, 2], data_format: 'channels_last' })
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

  it(`[convolutional.UpSampling2D.1] size 2x2 upsampling on 3x3x3 input, data_format='channels_first'`, function() {
    const key = `convolutional.UpSampling2D.1`
    console.log(`\n%c[${key}] size 2x2 upsampling on 3x3x3 input, data_format='channels_first'`, styles.h3)
    let testLayer = new layers.UpSampling2D({ size: [2, 2], data_format: 'channels_first' })
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

  it(`[convolutional.UpSampling2D.2] size 3x2 upsampling on 4x2x2 input, data_format='channels_last'`, function() {
    const key = `convolutional.UpSampling2D.2`
    console.log(`\n%c[${key}] size 3x2 upsampling on 4x2x2 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.UpSampling2D({ size: [3, 2], data_format: 'channels_last' })
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

  it(`[convolutional.UpSampling2D.3] size 1x3 upsampling on 4x3x2 input, data_format='channels_first'`, function() {
    const key = `convolutional.UpSampling2D.3`
    console.log(`\n%c[${key}] size 1x3 upsampling on 4x3x2 input, data_format='channels_first'`, styles.h3)
    let testLayer = new layers.UpSampling2D({ size: [1, 3], data_format: 'channels_first' })
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

  it(`[convolutional.UpSampling2D.4] size 2 upsampling on 1x3x2 input, data_format='channels_last'`, function() {
    const key = `convolutional.UpSampling2D.4`
    console.log(`\n%c[${key}] size 2 upsampling on 1x3x2 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.UpSampling2D({ size: 2, data_format: 'channels_last' })
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
