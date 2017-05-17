describe('pipeline_07', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = {
    inputShape: [8, 8, 2],
    layers: [
      {
        layerClass: 'Conv2D',
        attrs: {
          nbFilter: 4,
          nbRow: 3,
          nbCol: 3,
          activation: 'relu',
          borderMode: 'valid',
          subsample: [1, 1],
          dimOrdering: 'tf',
          bias: true
        }
      },
      {
        layerClass: 'MaxPooling2D',
        attrs: { poolSize: [2, 2], strides: [1, 1], borderMode: 'valid', dim_ordering: 'tf' }
      }
    ]
  }

  const key = 'pipeline_07'
  const title = `[${key}] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
  let modelLayers = []

  before(function() {
    console.log('\n%cpipeline_07', styles.h1)
    console.log(`\n%c${title}`, styles.h3)

    let weightsIndexOffset = 0
    for (let i = 0; i < testParams.layers.length; i++) {
      const layerConfig = testParams.layers[i]
      const attrs = Object.assign(layerConfig.attrs, { gpu: true, pipeline: true })
      const layerInstance = new layers[layerConfig.layerClass](attrs)
      const weightsArr = TEST_DATA[key].weights
        .slice(weightsIndexOffset, weightsIndexOffset + layerInstance.params.length)
        .map(w => new KerasJS.Tensor(w.data, w.shape))
      weightsIndexOffset += layerInstance.params.length
      layerInstance.setWeights(weightsArr)
      modelLayers.push(layerInstance)
    }

    // run dummy data once through to cache shape inference data, etc.
    let empty = new KerasJS.Tensor([], TEST_DATA[key].input.shape)
    for (let i = 0; i < testParams.layers.length; i++) {
      empty = modelLayers[i].call(empty)
    }
  })

  it(title, function() {
    let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
    console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
    const startTime = performance.now()
    for (let i = 0; i < testParams.layers.length; i++) {
      t = modelLayers[i].call(t)
    }
    t = modelLayers[testParams.layers.length - 1].transferFromPipeline(t)
    const endTime = performance.now()
    console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
    logTime(startTime, endTime)

    const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
    const shapeExpected = TEST_DATA[key].expected.shape
    assert.deepEqual(t.tensor.shape, shapeExpected)
    assert.isTrue(approxEquals(t.tensor, dataExpected))
  })
})
