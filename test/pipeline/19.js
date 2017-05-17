describe('pipeline_19', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = {
    inputShape: [3, 3],
    layers: [
      {
        layerClass: 'Activation',
        attrs: {
          activation: 'relu'
        }
      }
    ]
  }

  const key = 'pipeline_19'
  const title = `[${key}] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
  let modelLayers = []

  before(function() {
    console.log('\n%cpipeline_19', styles.h1)
    console.log(`\n%c${title}`, styles.h3)

    let weightsIndexOffset = 0
    for (let i = 0; i < testParams.layers.length; i++) {
      const layerConfig = testParams.layers[i]
      const attrs = Object.assign(layerConfig.attrs, { gpu: true, pipeline: true })
      const layerInstance = new layers[layerConfig.layerClass](attrs)
      modelLayers.push(layerInstance)
    }
  })

  it(title, function() {
    let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
    t.createWeblasTensor()
    t._fromPipeline = true
    t._actualShape = [3, 3]

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
