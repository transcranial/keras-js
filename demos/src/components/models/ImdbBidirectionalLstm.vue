<template>
  <div class="demo">
    <transition name="fade">
      <model-status v-if="modelLoading || modelInitializing" 
        :modelLoading="modelLoading"
        :modelLoadingProgress="modelLoadingProgress"
        :modelInitializing="modelInitializing"
        :modelInitProgress="modelInitProgress"
      ></model-status>
    </transition>
    <v-layout row wrap justify-center align-center>
      <v-flex sm12 md6 class="input-container elevation-1">
        <v-text-field
          :disabled="modelLoading || modelInitializing"
          label="input text"
          v-model="inputText"
          multi-line
          rows="10"
          color="primary"
          :spellcheck="false"
          @input.native="inputChanged"
        ></v-text-field>
        <div class="input-buttons">
          <v-btn :disabled="modelLoading || modelInitializing" flat bottom right color="primary" @click="randomSample">
            <v-icon left>add_circle</v-icon>LOAD SAMPLE TEXT
          </v-btn>
          <v-btn :disabled="modelLoading || modelInitializing" flat bottom right color="primary" @click="clear">
            <v-icon left>clear</v-icon>CLEAR
          </v-btn>
        </div>
      </v-flex>
      <v-flex sm12 md4 class="output-container">
        <div class="output-heading">Result:</div>
        <div class="output-value" :style="{ color: outputColor }">{{ Math.round(output[0] * 100) }}%</div>
        <div class="output-heading" v-if="isSampleText">
          <span>Actual label for sample text: </span>
          <span class="output-label" :class="sampleTextLabel">{{ sampleTextLabel }}</span>
        </div>
      </v-flex>
    </v-layout>
    <div class="architecture-container" v-if="!modelLoading">
      <div class="bg-line"></div>
      <div
        v-for="(layer, layerIndex) in architectureDiagramLayers"
        :key="`layer-${layerIndex}`"
        class="layer"
        :id="layer.name"
      >
        <div class="layer-class-name">{{ layer.className }}</div>
        <div class="layer-details"> {{ layer.details }}</div>
      </div>
    </div>
    <div class="lstm-visualization-container" v-if="!modelLoading && !modelRunning && inputTextParsed.length">
      <div
        v-for="(word, wordIndex) in inputTextParsed"
        :key="`word-${wordIndex}`"
        class="lstm-visualization-word"
        :style="{ background: stepwiseOutputColor[wordIndex] }"
      >{{ word }}</div>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
import axios from 'axios'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import ModelStatus from '../common/ModelStatus'

const MODEL_FILEPATH_PROD =
  'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.bin'
const MODEL_FILEPATH_DEV = '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.bin'

const ADDITIONAL_DATA_FILEPATHS_DEV = {
  wordIndex: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_test.json'
}
const ADDITIONAL_DATA_FILEPATHS_PROD = {
  wordIndex:
    'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict:
    'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_test.json'
}
const ADDITIONAL_DATA_FILEPATHS =
  process.env.NODE_ENV === 'production' ? ADDITIONAL_DATA_FILEPATHS_PROD : ADDITIONAL_DATA_FILEPATHS_DEV

const MAXLEN = 200

// start index, out-of-vocabulary index
// see https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3

// network layers
const ARCHITECTURE_DIAGRAM_LAYERS = [
  { name: 'embedding_1', className: 'Embedding', details: '200 time steps, dims 20000 -> 64' },
  {
    name: 'bidirectional_1',
    className: 'Bidirectional [LSTM]',
    details: '200 time steps, dims 64 -> 32, concat merge, tanh activation, hard sigmoid recurrent activation'
  },
  { name: 'dropout_1', className: 'Dropout', details: 'p=0.5 (active during training)' },
  { name: 'dense_1', className: 'Dense', details: 'output dims 1, sigmoid activation' }
]

export default {
  props: ['hasWebGL'],

  components: { ModelStatus },

  created() {
    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
      gpu: false
    })

    this.model.events.on('loadingProgress', this.handleLoadingProgress)
    this.model.events.on('initProgress', this.handleInitProgress)
  },

  async mounted() {
    await this.model.ready()
    this.loadAdditionalData()
  },

  beforeDestroy() {
    this.model.cleanup()
    this.model.events.removeAllListeners()
  },

  data() {
    return {
      useGPU: false,
      modelLoading: true,
      modelLoadingProgress: 0,
      modelInitializing: true,
      modelInitProgress: 0,
      modelRunning: false,
      input: new Float32Array(MAXLEN),
      output: new Float32Array(1),
      inputText: '',
      inputTextParsed: [],
      stepwiseOutput: [],
      wordIndex: {},
      wordDict: {},
      testSamples: [],
      isSampleText: false,
      sampleTextLabel: null,
      architectureDiagramLayers: ARCHITECTURE_DIAGRAM_LAYERS
    }
  },

  computed: {
    outputColor() {
      const p = this.output[0]
      if (p > 0 && p < 0.5) {
        return `rgba(242, 38, 19, ${1 - p})`
      } else if (p >= 0.5) {
        return `rgba(27, 188, 155, ${p})`
      }
      return '#69707a'
    },
    stepwiseOutputColor() {
      return this.stepwiseOutput.map(prob => {
        if (prob > 0 && prob < 0.5) {
          return `rgba(242, 38, 19, ${0.5 - prob})`
        } else if (prob >= 0.5) {
          return `rgba(27, 188, 155, ${prob - 0.5})`
        }
        return 'white'
      })
    }
  },

  methods: {
    handleLoadingProgress(progress) {
      this.modelLoadingProgress = Math.round(progress)
      if (progress === 100) {
        this.modelLoading = false
      }
    },
    handleInitProgress(progress) {
      this.modelInitProgress = Math.round(progress)
      if (progress === 100) {
        this.modelInitializing = false
      }
    },
    clear() {
      this.inputText = ''
      this.inputTextParsed = []
      this.output = new Float32Array(1)
      this.isSampleText = false
    },
    loadAdditionalData() {
      this.modelLoading = true
      const reqs = ['wordIndex', 'wordDict', 'testSamples'].map(key => {
        return axios.get(ADDITIONAL_DATA_FILEPATHS[key])
      })
      axios.all(reqs).then(
        axios.spread((wordIndex, wordDict, testSamples) => {
          this.wordIndex = wordIndex.data
          this.wordDict = wordDict.data
          this.testSamples = testSamples.data
          this.modelLoading = false
        })
      )
    },
    randomSample() {
      this.modelRunning = true
      this.isSampleText = true

      const randSampleIdx = _.random(0, this.testSamples.length - 1)
      const values = this.testSamples[randSampleIdx].values
      this.sampleTextLabel = this.testSamples[randSampleIdx].label === 0 ? 'negative' : 'positive'

      const words = values.map(idx => {
        if (idx === 0 || idx === 1) {
          return ''
        } else if (idx === 2) {
          return '<OOV>'
        } else {
          return this.wordDict[idx - INDEX_FROM]
        }
      })

      this.inputText = words.join(' ').trim()
      this.inputTextParsed = words.filter(w => !!w)

      this.input = new Float32Array(values)
      this.model.predict({ input: this.input }).then(outputData => {
        this.output = new Float32Array(outputData.output)
        this.stepwiseCalc()
        this.modelRunning = false
      })
    },
    inputChanged: _.debounce(function() {
      if (this.modelRunning) return
      if (this.inputText.trim() === '') {
        this.inputTextParsed = []
        return
      }

      this.modelRunning = true
      this.isSampleText = false

      this.inputTextParsed = this.inputText
        .trim()
        .toLowerCase()
        .split(/[\s.,!?]+/gi)

      this.input = new Float32Array(MAXLEN)
      // by convention, use 2 as OOV word
      // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
      // see https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
      let indices = this.inputTextParsed.map(word => {
        const index = this.wordIndex[word]
        return !index ? OOV_WORD_INDEX : index + INDEX_FROM
      })
      indices = [START_WORD_INDEX].concat(indices)
      indices = indices.slice(-MAXLEN)
      // padding and truncation (both pre sequence)
      const start = Math.max(0, MAXLEN - indices.length)
      for (let i = start; i < MAXLEN; i++) {
        this.input[i] = indices[i - start]
      }

      this.model.predict({ input: this.input }).then(outputData => {
        this.output = new Float32Array(outputData.output)
        this.stepwiseCalc()
        this.modelRunning = false
      })
    }, 200),
    stepwiseCalc() {
      const forwardHiddenStates = this.model.modelLayersMap.get('bidirectional_1').forwardLayer.hiddenStateSequence
      const backwardHiddenStates = this.model.modelLayersMap.get('bidirectional_1').backwardLayer.hiddenStateSequence
      const forwardDim = forwardHiddenStates.tensor.shape[1]
      const backwardDim = backwardHiddenStates.tensor.shape[1]

      const start = _.findIndex(this.input, idx => idx >= INDEX_FROM)
      if (start === -1) return

      const stepwiseOutput = []
      const tempTensor = ndarray(new Float32Array(forwardDim + backwardDim), [forwardDim + backwardDim])
      for (let i = start; i < MAXLEN; i++) {
        ops.assign(tempTensor.hi(forwardDim).lo(0), forwardHiddenStates.tensor.pick(i, null))
        ops.assign(
          tempTensor.hi(forwardDim + backwardDim).lo(forwardDim),
          backwardHiddenStates.tensor.pick(MAXLEN - i - 1, null)
        )
        stepwiseOutput.push(this.model.layerCall('dense_1', tempTensor).tensor.data[0])
      }
      this.stepwiseOutput = stepwiseOutput
    }
  }
}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.input-container {
  font-family: var(--font-monospace);
  background-color: whitesmoke;
  padding: 10px 30px;
  margin-bottom: 30px;

  & .input-buttons {
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }
}

.output-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  user-select: none;
  cursor: default;
  margin-bottom: 30px;

  & .output-heading {
    color: var(--color-lightgray);
    font-family: var(--font-monospace);
    font-size: 16px;
    text-align: center;
    margin: 10px;

    & span {
      display: block;
    }
    & span.output-label.positive {
      color: var(--color-green);
    }
    & span.output-label.negative {
      color: var(--color-red);
    }
  }

  & .output-value {
    transition: color 0.3s ease-in-out;
    font-family: var(--font-monospace);
    font-size: 42px;
    margin: 10px;
  }
}

.architecture-container {
  position: relative;
  width: 710px;
  margin: 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;

  & .bg-line {
    position: absolute;
    top: 50%;
    left: 0;
    height: 5px;
    width: 100%;
    background: whitesmoke;
  }

  & .layer {
    display: inline-block;
    width: 170px;
    margin-right: 10px;
    background: whitesmoke;
    border-radius: 5px;
    padding: 2px 10px 0px;
    z-index: 1;

    & .layer-class-name {
      color: var(--color-green);
      font-size: 14px;
      font-weight: bold;
    }

    & .layer-details {
      color: #999999;
      font-size: 10px;
      font-weight: bold;
    }
  }

  & .layer:last-child {
    margin-right: 0;
  }
}

.lstm-visualization-container {
  min-width: 700px;
  max-width: 900px;
  margin: 20px auto;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  border: 1px solid var(--color-lightgray);
  border-radius: 5px;
  padding: 10px;

  & .lstm-visualization-word {
    font-family: var(--font-monospace);
    font-size: 14px;
    color: var(--color-darkgray);
    padding: 3px 6px;
  }
}

/* vue transition `fade` */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}
</style>
