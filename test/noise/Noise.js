/* eslint-env browser, mocha */

describe('core layer: Noise', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  before(function () {
    console.log('\n%ccore layer: Noise', styles.h1)
  })

  it('[core.GaussianDropout.0] should just pass through tensor during test time', function () {
    const key = 'core.GaussianDropout.0'
    console.log(`\n%c[${key}] should pass through`, styles.h3)
    let testLayer1 = new layers.Dense({ outputDim: 2 })
    testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
    let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
    t = testLayer1.call(t)
    console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
    let testLayer2 = new layers.GaussianDropout({ p: 0.5 })
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

  it('[core.GaussianNoise.0] should just pass through tensor during test time', function () {
    const key = 'core.GaussianNoise.0'
    console.log(`\n%c[${key}] should pass through`, styles.h3)
    let testLayer1 = new layers.Dense({ outputDim: 2 })
    testLayer1.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
    let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
    t = testLayer1.call(t)
    console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
    let testLayer2 = new layers.GaussianNoise({ sigma: 0.5 })
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
