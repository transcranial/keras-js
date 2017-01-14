/* eslint-env browser, mocha */

describe('pipeline_18', function () {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  // layers ordered in same way as keras model config (see corresponding notebook)
  const testParams = {
    inputShape: [8, 8, 2],
    layers: [
      {
        name: 'conv_1-0',
        branch: 1,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 2, nbRow: 1, nbCol: 1, activation: 'relu', borderMode: 'valid', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        name: 'conv_1-1',
        branch: 1,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 4, nbRow: 3, nbCol: 3, activation: 'relu', borderMode: 'same', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        name: 'conv_2-0',
        branch: 2,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 5, nbRow: 3, nbCol: 3, activation: 'relu', borderMode: 'same', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        name: 'conv_0-0',
        branch: 0,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 4, nbRow: 3, nbCol: 3, activation: 'relu', borderMode: 'same', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        name: 'conv_1-2',
        branch: 1,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 2, nbRow: 1, nbCol: 1, activation: 'relu', borderMode: 'valid', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        name: 'conv_2-1',
        branch: 2,
        layerClass: 'Convolution2D',
        attrs: { nbFilter: 3, nbRow: 3, nbCol: 3, activation: 'relu', borderMode: 'same', subsample: [1, 1], dimOrdering: 'tf', bias: true }
      },
      {
        branches: [0, 1, 2],
        layerClass: 'Merge',
        attrs: { mode: 'concat' }
      }
    ]
  }

  const key = 'pipeline_18'
  const title = `[${key}] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
  let branch0 = []
  let branch1 = []
  let branch2 = []
  let mergeLayer

  before(function () {
    console.log('\n%cpipeline_18', styles.h1)
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
        branch0.push(layerInstance)
      } else if (layerConfig.branch === 1) {
        branch1.push(layerInstance)
      } else if (layerConfig.branch === 2) {
        branch2.push(layerInstance)
      } else {
        mergeLayer = layerInstance
      }
    }

    // run dummy data once through to cache shape inference data, etc.
    let empty = new KerasJS.Tensor([], TEST_DATA[key].inputs[0].shape)
    for (let i = 0; i < branch0.length; i++) {
      empty = branch0[i].call(empty)
    }
    empty = new KerasJS.Tensor([], TEST_DATA[key].inputs[1].shape)
    for (let i = 0; i < branch1.length; i++) {
      empty = branch1[i].call(empty)
    }
    empty = new KerasJS.Tensor([], TEST_DATA[key].inputs[2].shape)
    for (let i = 0; i < branch2.length; i++) {
      empty = branch2[i].call(empty)
    }
  })

  it(title, function () {
    let t0 = new KerasJS.Tensor(TEST_DATA[key].inputs[0].data, TEST_DATA[key].inputs[0].shape)
    let t1 = new KerasJS.Tensor(TEST_DATA[key].inputs[1].data, TEST_DATA[key].inputs[1].shape)
    let t2 = new KerasJS.Tensor(TEST_DATA[key].inputs[2].data, TEST_DATA[key].inputs[2].shape)
    console.log('%cin (branch 0)', styles.h4, stringifyCondensed(t0.tensor))
    console.log('%cin (branch 1)', styles.h4, stringifyCondensed(t1.tensor))
    console.log('%cin (branch 2)', styles.h4, stringifyCondensed(t2.tensor))
    const startTime = performance.now()
    for (let i = 0; i < branch0.length; i++) {
      t0 = branch0[i].call(t0)
    }
    for (let i = 0; i < branch1.length; i++) {
      t1 = branch1[i].call(t1)
    }
    for (let i = 0; i < branch2.length; i++) {
      t2 = branch2[i].call(t2)
    }
    let t = mergeLayer.call([t0, t1, t2])
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
