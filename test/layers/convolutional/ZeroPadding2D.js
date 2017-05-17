describe('convolutional layer: ZeroPadding2D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cconvolutional layer: ZeroPadding2D', styles.h1)
  })

  it(`[convolutional.ZeroPadding2D.0] padding (1,1) on 3x5x2 input, data_format='channels_last'`, function() {
    const key = `convolutional.ZeroPadding2D.0`
    console.log(`\n%c[${key}] padding (1,1) on 3x5x2 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: [1, 1], data_format: 'channels_last' })
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

  it(`[convolutional.ZeroPadding2D.1] padding (1,1) on 3x5x2 input, data_format='channels_first'`, function() {
    const key = `convolutional.ZeroPadding2D.1`
    console.log(`\n%c[${key}] padding (1,1) on 3x5x2 input, data_format='channels_first'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: [1, 1], data_format: 'channels_first' })
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

  it(`[convolutional.ZeroPadding2D.2] padding (3,2) on 2x6x4 input, data_format='channels_last'`, function() {
    const key = `convolutional.ZeroPadding2D.2`
    console.log(`\n%c[${key}] padding (3,2) on 2x6x4 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: [3, 2], data_format: 'channels_last' })
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

  it(`[convolutional.ZeroPadding2D.3] padding (3,2) on 2x6x4 input, data_format='channels_first'`, function() {
    const key = `convolutional.ZeroPadding2D.3`
    console.log(`\n%c[${key}] padding (3,2) on 2x6x4 input, data_format='channels_first'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: [3, 2], data_format: 'channels_first' })
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

  it(`[convolutional.ZeroPadding2D.4] padding ((1,2),(3,4)) on 2x6x4 input, data_format='channels_last'`, function() {
    const key = `convolutional.ZeroPadding2D.4`
    console.log(`\n%c[${key}] padding ((1,2),(3,4)) on 2x6x4 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: [[1, 2], [3, 4]], data_format: 'channels_last' })
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

  it(`[convolutional.ZeroPadding2D.5] padding 2 on 2x6x4 input, data_format='channels_last'`, function() {
    const key = `convolutional.ZeroPadding2D.5`
    console.log(`\n%c[${key}] padding 2 on 2x6x4 input, data_format='channels_last'`, styles.h3)
    let testLayer = new layers.ZeroPadding2D({ padding: 2, data_format: 'channels_last' })
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
