<template>
  <div class="demo imdb-bidirectional-lstm">
    <div class="title">
      <span>Bidirectional LSTM for IMDB sentiment classification</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column input-column">
        <div class="input-container">
          <mdl-textfield
            floating-label="input text"
            v-model="inputText"
            spellcheck="false"
            textarea
            rows="10"
            @input.native="inputChanged"
          ></mdl-textfield>
          <div class="input-buttons">
            <div class="input-load-button" @click="randomSample"><i class="material-icons">add_circle</i>LOAD SAMPLE TEXT</div>
            <div class="input-clear-button" @click="clear"><i class="material-icons">clear</i>CLEAR</div>
          </div>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <div class="output-heading">Result:</div>
          <div class="output-value" :style="{ color: outputColor }">{{ Math.round(output[0] * 100) }}%</div>
          <div class="output-heading" v-if="isSampleText">Actual label for sample text: <span class="output-label" :class="sampleTextLabel">{{ sampleTextLabel }}</span></div>
        </div>
      </div>
    </div>
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
import axios from 'axios'
import debounce from 'lodash/debounce'
import random from 'lodash/random'
import findIndex from 'lodash/findIndex'
import ops from 'ndarray-ops'
import Tensor from '../../../../src/Tensor'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

const ADDITIONAL_DATA_FILEPATHS_DEV = {
  wordIndex: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_test.json'
}
const ADDITIONAL_DATA_FILEPATHS_PROD = {
  wordIndex: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_test.json'
}
const ADDITIONAL_DATA_FILEPATHS = process.env.NODE_ENV === 'production'
  ? ADDITIONAL_DATA_FILEPATHS_PROD
  : ADDITIONAL_DATA_FILEPATHS_DEV

const MAXLEN = 200

// start index, out-of-vocabulary index
// see https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3

// network layers
const ARCHITECTURE_DIAGRAM_LAYERS = [
  { name: 'embedding_2', className: 'Embedding', details: '200 time steps, dims 20000 -> 64' },
  {
    name: 'bidirectional_2',
    className: 'Bidirectional [LSTM]',
    details: '200 time steps, dims 64 -> 32, concat merge, tanh activation, hard sigmoid recurrent activation'
  },
  { name: 'dropout_2', className: 'Dropout', details: 'p=0.5 (active during training)' },
  { name: 'dense_2', className: 'Dense', details: 'output dims 1, sigmoid activation' }
]

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: false,
      model: new KerasJS.Model(Object.assign({ gpu: false }, MODEL_CONFIG)), // eslint-disable-line
      modelLoading: true,
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
    loadingProgress: function() {
      return this.model.getLoadingProgress()
    },
    outputColor: function() {
      if (this.output[0] > 0 && this.output[0] < 0.5) {
        return `rgba(242, 38, 19, ${1 - this.output[0]})`
      } else if (this.output[0] >= 0.5) {
        return `rgba(27, 188, 155, ${this.output[0]})`
      }
      return '#69707a'
    },
    stepwiseOutputColor: function() {
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

  mounted: function() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.loadAdditionalData()
    })
  },

  methods: {
    clear: function() {
      this.inputText = ''
      this.inputTextParsed = []
      this.output = new Float32Array(1)
    },
    loadAdditionalData: function() {
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
    randomSample: function() {
      this.modelRunning = true
      this.isSampleText = true

      const randSampleIdx = random(0, this.testSamples.length - 1)
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
        this.output = outputData.output
        this.stepwiseCalc()
        this.modelRunning = false
      })
    },
    inputChanged: debounce(function() {
      if (this.modelRunning) return
      if (this.inputText.trim() === '') {
        this.inputTextParsed = []
        return
      }

      this.modelRunning = true
      this.isSampleText = false

      this.inputTextParsed = this.inputText.trim().toLowerCase().split(/[\s\.,!?]+/gi)

      this.input = new Float32Array(MAXLEN)
      // by convention, use 2 as OOV word
      // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
      // see https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
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
        this.output = outputData.output
        this.stepwiseCalc()
        this.modelRunning = false
      })
    }, 200),
    stepwiseCalc: function() {
      const fcLayer = this.model.modelLayersMap.get('dense_2')
      const forwardHiddenStates = this.model.modelLayersMap.get('bidirectional_2').forwardLayer.hiddenStateSequence
      const backwardHiddenStates = this.model.modelLayersMap.get('bidirectional_2').backwardLayer.hiddenStateSequence
      const forwardDim = forwardHiddenStates.tensor.shape[1]
      const backwardDim = backwardHiddenStates.tensor.shape[1]

      const start = findIndex(this.input, idx => idx >= INDEX_FROM)
      if (start === -1) return

      let stepwiseOutput = []
      for (let i = start; i < MAXLEN; i++) {
        let tempTensor = new Tensor([], [forwardDim + backwardDim])
        ops.assign(tempTensor.tensor.hi(forwardDim).lo(0), forwardHiddenStates.tensor.pick(i, null))
        ops.assign(
          tempTensor.tensor.hi(forwardDim + backwardDim).lo(forwardDim),
          backwardHiddenStates.tensor.pick(MAXLEN - i - 1, null)
        )
        stepwiseOutput.push(fcLayer.call(tempTensor).tensor.data[0])
      }
      this.stepwiseOutput = stepwiseOutput
    }
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.imdb-bidirectional-lstm {
  & .column {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  & .column.input-column {
    justify-content: center;

    & .input-container {
      text-align: right;
      margin: 5px 5px 5px 20px;
      position: relative;

      & .input-label {
        font-family: var(--font-cursive);
        font-size: 18px;
        color: var(--color-lightgray);
        text-align: left;
      }

      & .mdl-textfield {
        width: 550px;
      }

      & .input-buttons {
        display: flex;
        align-items: center;
        justify-content: space-between;

        & .input-load-button,
        & .input-clear-button {
          display: flex;
          align-items: center;
          color: var(--color-lightgray);
          transition: color 0.2s ease-in;

          & .material-icons {
            margin-right: 5px;
          }

          &:hover {
            color: var(--color-green-lighter);
            cursor: pointer;
          }
        }

        & .input-load-button {
          &:hover { color: var(--color-green); }
        }
      }
    }
  }

  & .column.output-column {
    justify-content: center;

    & .output {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      user-select: none;
      cursor: default;

      & .output-heading {
        max-width: 200px;
        color: var(--color-lightgray);
        font-family: var(--font-monospace);
        font-size: 16px;
        margin: 10px;

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
  }

  & .architecture-container {
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

  & .lstm-visualization-container {
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
}
</style>
