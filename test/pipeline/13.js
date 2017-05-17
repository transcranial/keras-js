describe('pipeline_13', function() {
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
        branch: 0,
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
        branch: 1,
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
      { branches: [0, 1], layerClass: 'Merge', attrs: { mode: 'sum' } }
    ]
  }

  const key = 'pipeline_13'
  const title = `[${key}] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
  let branch_0 = []
  let branch_1 = []
  let mergeLayer

  before(function() {
    console.log('\n%cpipeline_13', styles.h1)
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
      if (layerConfig.branch === 0) {
        branch_0.push(layerInstance)
      } else if (layerConfig.branch === 1) {
        branch_1.push(layerInstance)
      } else {
        mergeLayer = layerInstance
      }
    }

    // run dummy data once through to cache shape inference data, etc.
    let empty = new KerasJS.Tensor([], TEST_DATA[key].inputs[0].shape)
    for (let i = 0; i < branch_0.length; i++) {
      empty = branch_0[i].call(empty)
    }
    empty = new KerasJS.Tensor([], TEST_DATA[key].inputs[1].shape)
    for (let i = 0; i < branch_1.length; i++) {
      empty = branch_1[i].call(empty)
    }
  })

  it(title, function() {
    let t_0 = new KerasJS.Tensor(TEST_DATA[key].inputs[0].data, TEST_DATA[key].inputs[0].shape)
    let t_1 = new KerasJS.Tensor(TEST_DATA[key].inputs[1].data, TEST_DATA[key].inputs[1].shape)
    console.log('%cin (branch 0)', styles.h4, stringifyCondensed(t_0.tensor))
    console.log('%cin (branch 1)', styles.h4, stringifyCondensed(t_1.tensor))
    const startTime = performance.now()
    for (let i = 0; i < branch_0.length; i++) {
      t_0 = branch_0[i].call(t_0)
    }
    for (let i = 0; i < branch_1.length; i++) {
      t_1 = branch_1[i].call(t_1)
    }
    let t = mergeLayer.call([t_0, t_1])
    t = mergeLayer.transferFromPipeline(t)
    const endTime = performance.now()
    console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
    logTime(startTime, endTime)

    const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
    const shapeExpected = TEST_DATA[key].expected.shape
    assert.deepEqual(t.tensor.shape, shapeExpected)
    assert.isTrue(approxEquals(t.tensor, dataExpected))
  })
})
