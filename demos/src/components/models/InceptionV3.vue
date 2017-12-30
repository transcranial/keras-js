<template>
  <div class="demo">
    <imagenet
      modelName="inception_v3"
      :modelFilepath="modelFilepath"
      :hasWebGL="hasWebGL"
      :imageSize="299"
      :visualizations="['CAM']"
      :preprocess="preprocess"
    ></imagenet>
  </div>
</template>

<script>
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import Imagenet from '../common/Imagenet'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/inception_v3/inception_v3.bin'
const MODEL_FILEPATH_DEV = '/demos/data/inception_v3/inception_v3.bin'

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

      ops.divseq(dataTensor, 127.5)
      ops.subseq(dataTensor, 1)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const preprocessedData = dataProcessedTensor.data
      return preprocessedData
    }
  }
}
</script>

<style scoped lang="postcss">

</style>
