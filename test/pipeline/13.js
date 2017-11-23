describe('pipeline_13', function() {
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
        attrs: { name: 'layer_0', filters: 2, kernel_size: 3, strides: 1, padding: 'valid' },
        inbound: [],
        outbound: ['layer_1']
      },
      {
        layerClass: 'Activation',
        attrs: { name: 'layer_1', activation: 'relu' },
        inbound: ['layer_0'],
        outbound: ['layer_2']
      },
      {
        layerClass: 'Conv2D',
        attrs: { name: 'layer_2', filters: 2, kernel_size: 3, strides: 1, padding: 'valid' },
        inbound: ['layer_1'],
        outbound: ['layer_3']
      },
      {
        layerClass: 'Activation',
        attrs: { name: 'layer_3', activation: 'relu' },
        inbound: ['layer_2'],
        outbound: ['layer_4']
      },
      {
        layerClass: 'MaxPooling2D',
        attrs: { name: 'layer_4', pool_size: [2, 2], strides: [1, 1] },
        inbound: ['layer_3'],
        outbound: ['layer_5']
      },
      {
        layerClass: 'Dropout',
        attrs: { name: 'layer_5', rate: 0.25 },
        inbound: ['layer_4'],
        outbound: ['layer_6']
      },
      {
        layerClass: 'Flatten',
        attrs: { name: 'layer_6' },
        inbound: ['layer_5'],
        outbound: ['layer_7']
      },
      {
        layerClass: 'Dense',
        attrs: { name: 'layer_7', units: 3 },
        inbound: ['layer_6'],
        outbound: ['layer_8']
      },
      {
        layerClass: 'Activation',
        attrs: { name: 'layer_8', activation: 'relu' },
        inbound: ['layer_7'],
        outbound: ['layer_9']
      },
      {
        layerClass: 'Dropout',
        attrs: { name: 'layer_9', rate: 0.5 },
        inbound: ['layer_8'],
        outbound: ['layer_10']
      },
      {
        layerClass: 'Dense',
        attrs: { name: 'layer_10', units: 3 },
        inbound: ['layer_9'],
        outbound: ['layer_11']
      },
      {
        layerClass: 'Activation',
        attrs: { name: 'layer_11', activation: 'softmax' },
        inbound: ['layer_10'],
        outbound: []
      }
    ]
  }

  const key = 'pipeline_13'

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
