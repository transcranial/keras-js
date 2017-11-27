describe('graph_07', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = {
    layers: [
      {
        branch: 0,
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_0_0',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: [],
        outbound: ['layer_0_1']
      },
      {
        branch: 1,
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_1_0',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: [],
        outbound: ['layer_1_1']
      },
      {
        branch: 0,
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_0_1',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: ['layer_0_0'],
        outbound: ['layer_2_0']
      },
      {
        branch: 1,
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_1_1',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: ['layer_1_0'],
        outbound: ['layer_2_0']
      },
      {
        branch: 2,
        layerClass: 'Concatenate',
        attrs: { name: 'layer_2_0' },
        inbound: ['layer_0_1', 'layer_1_1'],
        outbound: ['layer_2_1']
      },
      {
        branch: 2,
        layerClass: 'Conv2D',
        attrs: {
          name: 'layer_2_1',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'linear',
          use_bias: true
        },
        inbound: ['layer_2_0'],
        outbound: []
      }
    ]
  }

  const key = 'graph_07'

  before(function() {
    console.log(`\n%c${key}`, styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    const title = `[CPU] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
    const branch_0 = []
    const branch_1 = []
    const branch_2 = []

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
        if (layerConfig.branch === 0) {
          branch_0.push(layerInstance)
        } else if (layerConfig.branch === 1) {
          branch_1.push(layerInstance)
        } else if (layerConfig.branch === 2) {
          branch_2.push(layerInstance)
        }
      }

      // run dummy data once through to cache shape inference data, etc.
      let empty_0 = new KerasJS.Tensor([], TEST_DATA[key].inputs[0].shape)
      for (let i = 0; i < branch_0.length; i++) {
        empty_0 = branch_0[i].call(empty_0)
      }
      let empty_1 = new KerasJS.Tensor([], TEST_DATA[key].inputs[1].shape)
      for (let i = 0; i < branch_1.length; i++) {
        empty_1 = branch_1[i].call(empty_1)
      }
      let empty_2 = branch_2[0].call([empty_0, empty_1])
      for (let i = 1; i < branch_2.length; i++) {
        empty_2 = branch_2[i].call(empty_2)
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
      let t = branch_2[0].call([t_0, t_1])
      for (let i = 1; i < branch_2.length; i++) {
        t = branch_2[i].call(t)
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
    const branch_0 = []
    const branch_1 = []
    const branch_2 = []

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
        if (layerConfig.branch === 0) {
          branch_0.push(layerInstance)
        } else if (layerConfig.branch === 1) {
          branch_1.push(layerInstance)
        } else if (layerConfig.branch === 2) {
          branch_2.push(layerInstance)
        }
      }

      // run dummy data once through to cache shape inference data, etc.
      let empty_0 = new KerasJS.Tensor([], TEST_DATA[key].inputs[0].shape)
      for (let i = 0; i < branch_0.length; i++) {
        empty_0 = branch_0[i].call(empty_0)
      }
      let empty_1 = new KerasJS.Tensor([], TEST_DATA[key].inputs[1].shape)
      for (let i = 0; i < branch_1.length; i++) {
        empty_1 = branch_1[i].call(empty_1)
      }
      let empty_2 = branch_2[0].call([empty_0, empty_1])
      for (let i = 1; i < branch_2.length; i++) {
        empty_2 = branch_2[i].call(empty_2)
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
      let t = branch_2[0].call([t_0, t_1])
      for (let i = 1; i < branch_2.length; i++) {
        t = branch_2[i].call(t)
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
