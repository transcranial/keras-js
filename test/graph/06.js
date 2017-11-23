describe('graph_06', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = {
    layers: [
      {
        branch: 1,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_1_0',
          filters: 2,
          kernel_size: 1,
          strides: 1,
          padding: 'valid',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: [],
        outbound: ['conv_1_1']
      },
      {
        branch: 1,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_1_1',
          filters: 4,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: ['conv_1_0'],
        outbound: ['conv_1_2']
      },
      {
        branch: 2,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_2_0',
          filters: 5,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: [],
        outbound: ['conv_2_1']
      },
      {
        branch: 0,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_0_0',
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
        outbound: ['concatenate_1']
      },
      {
        branch: 1,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_1_2',
          filters: 2,
          kernel_size: 1,
          strides: 1,
          padding: 'valid',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: ['conv_1_1'],
        outbound: ['concatenate_1']
      },
      {
        branch: 2,
        layerClass: 'Conv2D',
        attrs: {
          name: 'conv_2_1',
          filters: 3,
          kernel_size: 3,
          strides: 1,
          padding: 'same',
          data_format: 'channels_last',
          dilation_rate: 1,
          activation: 'relu',
          use_bias: true
        },
        inbound: ['conv_2_0'],
        outbound: ['concatenate_1']
      },
      {
        branch: [0, 1, 2],
        layerClass: 'Concatenate',
        attrs: { name: 'concatenate_1' },
        inbound: ['conv_0_0', 'conv_1_2', 'conv_2_1'],
        outbound: []
      }
    ]
  }

  const key = 'graph_06'

  before(function() {
    console.log(`\n%c${key}`, styles.h1)
  })

  /*********************************************************
   * CPU
   *********************************************************/
  describe('CPU', function() {
    const title = `[CPU] ${testParams.layers.map(layer => layer.layerClass).join('-')}`
    let branch_0 = []
    let branch_1 = []
    let branch_2 = []
    let mergeLayer

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
        } else {
          mergeLayer = layerInstance
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
      let empty_2 = new KerasJS.Tensor([], TEST_DATA[key].inputs[2].shape)
      for (let i = 0; i < branch_2.length; i++) {
        empty_2 = branch_2[i].call(empty_2)
      }
      let empty = mergeLayer.call([empty_0, empty_1, empty_2])
    })

    it(title, function() {
      let t_0 = new KerasJS.Tensor(TEST_DATA[key].inputs[0].data, TEST_DATA[key].inputs[0].shape)
      let t_1 = new KerasJS.Tensor(TEST_DATA[key].inputs[1].data, TEST_DATA[key].inputs[1].shape)
      let t_2 = new KerasJS.Tensor(TEST_DATA[key].inputs[2].data, TEST_DATA[key].inputs[2].shape)
      console.log('%cin (branch 0)', styles.h4, stringifyCondensed(t_0.tensor))
      console.log('%cin (branch 1)', styles.h4, stringifyCondensed(t_1.tensor))
      console.log('%cin (branch 2)', styles.h4, stringifyCondensed(t_2.tensor))
      const startTime = performance.now()
      for (let i = 0; i < branch_0.length; i++) {
        t_0 = branch_0[i].call(t_0)
      }
      for (let i = 0; i < branch_1.length; i++) {
        t_1 = branch_1[i].call(t_1)
      }
      for (let i = 0; i < branch_2.length; i++) {
        t_2 = branch_2[i].call(t_2)
      }
      let t = mergeLayer.call([t_0, t_1, t_2])
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
    let branch_0 = []
    let branch_1 = []
    let branch_2 = []
    let mergeLayer

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
        } else {
          mergeLayer = layerInstance
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
      let empty_2 = new KerasJS.Tensor([], TEST_DATA[key].inputs[2].shape)
      for (let i = 0; i < branch_2.length; i++) {
        empty_2 = branch_2[i].call(empty_2)
      }
      let empty = mergeLayer.call([empty_0, empty_1, empty_2])
    })

    it(title, function() {
      let t_0 = new KerasJS.Tensor(TEST_DATA[key].inputs[0].data, TEST_DATA[key].inputs[0].shape)
      let t_1 = new KerasJS.Tensor(TEST_DATA[key].inputs[1].data, TEST_DATA[key].inputs[1].shape)
      let t_2 = new KerasJS.Tensor(TEST_DATA[key].inputs[2].data, TEST_DATA[key].inputs[2].shape)
      console.log('%cin (branch 0)', styles.h4, stringifyCondensed(t_0.tensor))
      console.log('%cin (branch 1)', styles.h4, stringifyCondensed(t_1.tensor))
      console.log('%cin (branch 2)', styles.h4, stringifyCondensed(t_2.tensor))
      const startTime = performance.now()
      for (let i = 0; i < branch_0.length; i++) {
        t_0 = branch_0[i].call(t_0)
      }
      for (let i = 0; i < branch_1.length; i++) {
        t_1 = branch_1[i].call(t_1)
      }
      for (let i = 0; i < branch_2.length; i++) {
        t_2 = branch_2[i].call(t_2)
      }
      let t = mergeLayer.call([t_0, t_1, t_2])
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
