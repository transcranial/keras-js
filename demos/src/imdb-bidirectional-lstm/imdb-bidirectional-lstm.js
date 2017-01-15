import './imdb-bidirectional-lstm.css';

import axios from 'axios';
import debounce from 'lodash/debounce';
import random from 'lodash/random';
import findIndex from 'lodash/findIndex';
import ops from 'ndarray-ops';
import Tensor from '../../../src/Tensor';

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: '/demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
};
const MODEL_FILEPATHS_PROD = {
  model: 'demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: 'demos/data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
};
const MODEL_CONFIG = {
  filepaths: process.env.NODE_ENV === 'production'
    ? MODEL_FILEPATHS_PROD
    : MODEL_FILEPATHS_DEV
};

const ADDITIONAL_DATA_FILEPATHS_DEV = {
  wordIndex: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: '/demos/data/imdb_bidirectional_lstm/imdb_dataset_test.json'
};
const ADDITIONAL_DATA_FILEPATHS_PROD = {
  wordIndex: 'demos/data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
  wordDict: 'demos/data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
  testSamples: 'demos/data/imdb_bidirectional_lstm/imdb_dataset_test.json'
};
const ADDITIONAL_DATA_FILEPATHS = process.env.NODE_ENV === 'production'
  ? ADDITIONAL_DATA_FILEPATHS_PROD
  : ADDITIONAL_DATA_FILEPATHS_DEV;

const MAXLEN = 200;

// start index, out-of-vocabulary index
// see https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
const START_WORD_INDEX = 1;
const OOV_WORD_INDEX = 2;
const INDEX_FROM = 3;

// network layers
const ARCHITECTURE_DIAGRAM_LAYERS = [
  {
    name: 'embedding_1',
    className: 'Embedding',
    details: '200 time steps, dim 20000 -> 64'
  },
  {
    name: 'bidirectional_1',
    className: 'Bidirectional [LSTM]',
    details: '200 time steps, dim 64 -> 32, concat merge, tanh activation, hard sigmoid inner activation'
  },
  {
    name: 'dropout_1',
    className: 'Dropout',
    details: 'p=0.5 (active during training)'
  },
  {
    name: 'dense_1',
    className: 'Dense',
    details: 'output dim 1, sigmoid activation'
  }
];

/**
 *
 * VUE COMPONENT
 *
 */
export const ImdbBidirectionalLstm = Vue.extend({
  props: [ 'hasWebgl' ],
  template: require('raw-loader!./imdb-bidirectional-lstm.template.html'),
  data: function() {
    return {
      showInfoPanel: true,
      useGpu: false,
      model: new KerasJS.Model(Object.assign({ gpu: false }, MODEL_CONFIG)),
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
    };
  },
  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress();
    },
    outputColor: function() {
      if (this.output[0] > 0 && this.output[0] < 0.5) {
        return `rgba(242, 38, 19, ${1 - this.output[0]})`;
      } else if (this.output[0] >= 0.5) {
        return `rgba(27, 188, 155, ${this.output[0]})`;
      }
      return '#69707a';
    },
    stepwiseOutputColor: function() {
      return this.stepwiseOutput.map(prob => {
        if (prob > 0 && prob < 0.5) {
          return `rgba(242, 38, 19, ${0.5 - prob})`;
        } else if (prob >= 0.5) {
          return `rgba(27, 188, 155, ${prob - 0.5})`;
        }
        return 'white';
      });
    }
  },
  ready: function() {
    this.model.ready().then(() => {
      this.modelLoading = false;
      this.loadAdditionalData();
    });
  },
  methods: {
    closeInfoPanel: function() {
      this.showInfoPanel = false;
    },
    toggleGpu: function() {
      this.model.toggleGpu(!this.useGpu);
    },
    clear: function(e) {
      this.inputText = '';
      this.inputTextParsed = [];
      this.output = new Float32Array(1);
    },
    loadAdditionalData: function() {
      this.modelLoading = true;
      const reqs = [ 'wordIndex', 'wordDict', 'testSamples' ].map(key => {
        return axios.get(ADDITIONAL_DATA_FILEPATHS[key]);
      });
      axios.all(reqs).then(
        axios.spread((wordIndex, wordDict, testSamples) => {
          this.wordIndex = wordIndex.data;
          this.wordDict = wordDict.data;
          this.testSamples = testSamples.data;
          this.modelLoading = false;
        })
      );
    },
    randomSample: function() {
      this.modelRunning = true;
      this.isSampleText = true;

      const randSampleIdx = random(0, this.testSamples.length - 1);
      const values = this.testSamples[randSampleIdx].values;
      this.sampleTextLabel = this.testSamples[randSampleIdx].label === 0
        ? 'negative'
        : 'positive';

      const words = values.map(idx => {
        if (idx === 0 || idx === 1) {
          return '';
        } else if (idx === 2) {
          return '<OOV>';
        } else {
          return this.wordDict[idx - INDEX_FROM];
        }
      });

      this.inputText = words.join(' ').trim();
      this.inputTextParsed = words.filter(w => !!w);

      this.input = new Float32Array(values);
      this.model.predict({ input: this.input }).then(outputData => {
        this.output = outputData.output;
        this.stepwiseCalc();
        this.modelRunning = false;
      });
    },
    inputChanged: debounce(
      function() {
        if (this.modelRunning)
          return;
        if (this.inputText.trim() === '') {
          this.inputTextParsed = [];
          return;
        }

        this.modelRunning = true;
        this.isSampleText = false;

        this.inputTextParsed = this.inputText
          .trim()
          .toLowerCase()
          .split(/[\s\.,!?]+/ig);

        this.input = new Float32Array(MAXLEN);
        // by convention, use 2 as OOV word
        // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
        // see https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
        let indices = this.inputTextParsed.map(word => {
          const index = this.wordIndex[word];
          return !index ? OOV_WORD_INDEX : index + INDEX_FROM;
        });
        indices = [ START_WORD_INDEX ].concat(indices);
        indices = indices.slice(-MAXLEN);
        // padding and truncation (both pre sequence)
        const start = Math.max(0, MAXLEN - indices.length);
        for (let i = start; i < MAXLEN; i++) {
          this.input[i] = indices[i - start];
        }

        this.model.predict({ input: this.input }).then(outputData => {
          this.output = outputData.output;
          this.stepwiseCalc();
          this.modelRunning = false;
        });
      },
      200
    ),
    stepwiseCalc: function() {
      const fcLayer = this.model.modelLayersMap.get('dense_1');
      const forwardHiddenStates = this.model.modelLayersMap.get(
        'bidirectional_1'
      ).forwardLayer.hiddenStateSequence;
      const backwardHiddenStates = this.model.modelLayersMap.get(
        'bidirectional_1'
      ).backwardLayer.hiddenStateSequence;
      const forwardDim = forwardHiddenStates.tensor.shape[1];
      const backwardDim = backwardHiddenStates.tensor.shape[1];

      const start = findIndex(this.input, idx => idx >= INDEX_FROM);
      if (start === -1)
        return;

      let stepwiseOutput = [];
      for (let i = start; i < MAXLEN; i++) {
        let tempTensor = new Tensor([], [ forwardDim + backwardDim ]);
        ops.assign(
          tempTensor.tensor.hi(forwardDim).lo(0),
          forwardHiddenStates.tensor.pick(i, null)
        );
        ops.assign(
          tempTensor.tensor.hi(forwardDim + backwardDim).lo(forwardDim),
          backwardHiddenStates.tensor.pick(MAXLEN - i - 1, null)
        );
        stepwiseOutput.push(fcLayer.call(tempTensor).tensor.data[0]);
      }
      this.stepwiseOutput = stepwiseOutput;
    }
  }
});
