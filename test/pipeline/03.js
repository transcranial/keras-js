describe('pipeline_03', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = {
    layers: [
      {
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_0',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'valid',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: [],
        outbound: ['layer_1']
      },
      {
        layerClass: 'BatchNormalization',
        attrs: { name: 'layer_1', epsilon: 1e-5, axis: -1, center: true, scale: true },
        inbound: ['layer_0'],
        outbound: []
      }
    ]
  }

  const key = 'pipeline_03'

  before(function() {
    console.log(`\n%c${key}`, styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    const title = `[CPU] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
    let modelLayers = []

    before(function() {
      console.log('\n%cCPU', styles.h2)
      console.log(`\n%c${title}`, styles.h3)

      let weightsIndexOffset = 0
      for (let i = 0; i < testParams.layers.length; i++) {
        const layerConfig = testParams.layers[i]
        const attrs = Object.assign(layerConfig.attrs)
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
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })

  /*********************************************************
   * GPU
   *********************************************************/
  describe('GPU', function() {
    const title = `[GPU] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
    let modelLayers = []

    before(function() {
      console.log('\n%cGPU', styles.h2)
      console.log(`\n%c${title}`, styles.h3)

      let weightsIndexOffset = 0
      for (let i = 0; i < testParams.layers.length; i++) {
        const layerConfig = testParams.layers[i]
        const layerInstance = new layers[layerConfig.layerClass](Object.assign(layerConfig.attrs, { gpu: true }))
        const weightsArr = TEST_DATA[key].weights
          .slice(weightsIndexOffset, weightsIndexOffset + layerInstance.params.length)
          .map(w => new KerasJS.Tensor(w.data, w.shape))
        weightsIndexOffset += layerInstance.params.length
        layerInstance.setWeights(weightsArr)
        layerInstance.inbound = layerConfig.inbound
        layerInstance.outbound = layerConfig.outbound
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
