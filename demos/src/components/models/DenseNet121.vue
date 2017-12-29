<template>
  <div class="demo">
    <imagenet
      modelName="densenet121"
      :modelFilepath="modelFilepath"
      :hasWebGL="hasWebGL"
      :imageSize="224"
      :visualizations="['CAM']"
      :preprocess="preprocess"
    ></imagenet>
  </div>
</template>

<script>
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import Imagenet from '../common/Imagenet'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/densenet121/densenet121.bin'
const MODEL_FILEPATH_DEV = '/demos/data/densenet121/densenet121.bin'

export default {
  props: ['hasWebGL'],

  components: { Imagenet },

  data() {
    return {
      modelFilepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV
    }
  },

  methods: {
    preprocess(imageData) {
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
      const dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])

      ops.subseq(dataTensor.pick(null, null, 2), 103.939)
      ops.subseq(dataTensor.pick(null, null, 1), 116.779)
      ops.subseq(dataTensor.pick(null, null, 0), 123.68)
      ops.mulseq(dataTensor, 0.017)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))

      const preprocessedData = dataProcessedTensor.data
      return preprocessedData
    }
  }
}
</script>

<style scoped lang="postcss">

</style>
