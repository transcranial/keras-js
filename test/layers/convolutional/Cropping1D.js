describe('convolutional layer: Cropping1D', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function() {
    console.log('\n%cconvolutional layer: Cropping1D', styles.h1)
  })

  it(`[convolutional.Cropping1D.0] cropping (1,1) on 6x4 input`, function() {
    const key = `convolutional.Cropping1D.0`
    console.log(`\n%c[${key}] cropping (1,1) on 6x4 input`, styles.h3)
    let testLayer = new layers.Cropping1D({ cropping: [1, 1] })
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

  it(`[convolutional.Cropping1D.1] cropping (2,3) on 6x4 input`, function() {
    const key = `convolutional.Cropping1D.1`
    console.log(`\n%c[${key}] cropping (2,3) on 6x4 input`, styles.h3)
    let testLayer = new layers.Cropping1D({ cropping: [2, 3] })
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

  it(`[convolutional.Cropping1D.2] cropping 2 on 6x4 input`, function() {
    const key = `convolutional.Cropping1D.2`
    console.log(`\n%c[${key}] cropping 2 on 6x4 input`, styles.h3)
    let testLayer = new layers.Cropping1D({ cropping: 2 })
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
