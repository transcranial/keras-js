/* global XMLHttpRequest */
import toPairs from 'lodash/toPairs'
import mapKeys from 'lodash/mapKeys'
import camelCase from 'lodash/camelCase'
import find from 'lodash/find'
import keys from 'lodash/keys'
import values from 'lodash/values'
import sum from 'lodash/sum'
import isEqual from 'lodash/isEqual'
import every from 'lodash/every'
import * as layers from './layers'
import Input from './Input'
import Tensor from './Tensor'

/**
 * Model class
 */
export default class Model {
  /**
   * create new Model class
   * @param {object} filepaths
   * @param {string} filepaths.modelFilepath - path to model architecture configuration (json)
   * @param {string} filepaths.weightsFilepath - path to weights data (arraybuffer)
   * @param {string} filepaths.metadataFilepath - path to weights metadata (json)
   * @param {object} [config]
   * @param {object} [config.headers] - any additional HTTP headers required for resource fetching
   */
  constructor (filepaths, config = {}) {
    if (!filepaths.model || !filepaths.weights || !filepaths.metadata) {
      throw new Error('File paths must be declared for model, weights, and metadata.')
    }
    this.filepaths = filepaths
    this.filetypes = {
      model: 'json',
      weights: 'arraybuffer',
      metadata: 'json'
    }

    this.config = config

    this.data = {
      // object representing the model architecture configuration,
      // directly from the to_json() method in Keras
      model: {},
      // ArrayBuffer of all the weights, sequentially concatenated
      // see encoder.py for construction details - essentially the raw flattened
      // numerical data from the HDF5 file is extracted sequentially and concatenated.
      weights: null,
      // array of weight tensor metadata, used to reconstruct tensors from the raw
      // weights ArrayBuffer above.
      metadata: []
    }

    // keep track of XHR requests
    this.xhrs = {
      model: null,
      weights: null,
      metadata: null
    }

    // XHR progress
    this.xhrProgress = {
      model: 0,
      weights: 0,
      metadata: 0
    }

    // map of model layers
    this.modelLayersMap = new Map()

    // directed acyclic graph of model network
    this.modelDAG = {}

    // input tensors
    this.inputTensors = {}

    this._ready = this.initialize()
  }

  /**
   * Promise for when model data is loaded and layers are initialized.
   * @returns {Promise}
   */
  ready () {
    return this._ready
  }

  /**
   * Cancels any existing XHR requests
   */
  interrupt () {
    const dataTypes = ['model', 'weights', 'metdata']
    dataTypes.forEach(type => {
      if (this.xhrs[type]) {
        this.xhrs[type].abort()
        this.xhrs[type] = null
      }
    })
  }

  /**
   * Model initialization
   * @returns {Promise}
   */
  initialize () {
    const dataTypes = ['model', 'weights', 'metadata']
    return Promise.all(
      dataTypes.map(type => this.dataRequest(type, this.config.headers))
    )
    .then(() => {
      this.createLayers()
      return Promise.resolve()
    })
    .catch(err => {
      console.log(err)
      this.interrupt()
    })
  }

  /**
   * Makes XHR request
   * @returns {Promise}
   */
  dataRequest (type, headers = {}) {
    return new Promise((resolve, reject) => {
      let xhr = new XMLHttpRequest()
      xhr.open('GET', this.filepaths[type], true)
      xhr.responseType = this.filetypes[type]
      for (const [h, v] of toPairs(headers)) {
        xhr.setRequestHeader(h, v)
      }
      xhr.onload = e => {
        this.data[type] = xhr.response
        this.xhrs[type] = null
        this.xhrProgress[type] = 100
        resolve()
      }
      xhr.onprogress = e => {
        if (e.lengthComputable) {
          const percentComplete = Math.round(100 * e.loaded / e.total)
          this.xhrProgress[type] = percentComplete
        }
      }
      xhr.onerror = e => reject(e)
      xhr.send(null)
      this.xhrs[type] = xhr
    })
  }

  /**
   * Loading progress calculated from all the XHRs combined.
   */
  getLoadingProgress () {
    const progressValues = values(this.xhrProgress)
    return Math.round(sum(progressValues) / progressValues.length)
  }

  /**
   * Builds network layer DAG
   * For Keras models of class Sequential, we still convert the list into DAG format
   * for straightforward interoperability with graph models.
   *
   * Layer constructors take an `attrs` object, which contain layer parameters among
   * other information. Note that in the Keras model config object variables are
   * in snake_case. We convert the variable names to camelCase here.
   */
  createLayers () {
    const modelClass = this.data.model.class_name

    if (modelClass === 'Sequential') {
      const modelConfig = this.data.model.config
      const inputName = 'input'

      modelConfig.forEach((layerDef, index) => {
        const layerClass = layerDef.class_name
        const layerConfig = layerDef.config

        // create Input node at start
        if (index === 0) {
          const inputShape = layerConfig.batch_input_shape.slice(1)
          const layer = new Input({
            name: inputName,
            inputShape
          })
          this.modelLayersMap.set(inputName, layer)
          this.modelDAG[inputName] = {
            name: inputName,
            outbound: []
          }
          this.inputTensors[inputName] = new Tensor([], inputShape)
        }

        if (layerClass in layers) {
          const attrs = mapKeys(layerConfig, (v, k) => camelCase(k))
          const layer = new layers[layerClass](attrs)

          // layer weights
          if (layer.params && layer.params.length) {
            const weights = layer.params.map(param => {
              const paramMetadata = find(this.data.metadata, meta => {
                const weightRE = new RegExp(`^${layerConfig.name}_${param}`)
                return meta.layer_name === layerConfig.name &&
                  weightRE.test(meta.weight_name)
              })
              if (!paramMetadata) {
                throw new Error(`[Model] error loading weights.`)
              }

              const { offset, length, shape } = paramMetadata
              return new Tensor(new Float32Array(this.data.weights, offset, length), shape)
            })
            layer.setWeights(weights)
          }

          this.modelLayersMap.set(layerConfig.name, layer)
          this.modelDAG[layerConfig.name] = {
            name: layerConfig.name,
            outbound: []
          }
          if (index === 0) {
            this.modelDAG[inputName].outbound.push(layerConfig.name)
          } else {
            const prevLayerConfig = modelConfig[index - 1].config
            this.modelDAG[prevLayerConfig.name].outbound.push(layerConfig.name)
          }
        } else {
          throw new Error(`Layer ${layerClass} specified in model configuration is not implemented!`)
        }
      })
    }
  }

  /**
   * Predict API
   */
  predict (inputData) {
    const inputNames = keys(this.inputTensors)
    if (!isEqual(keys(inputData), inputNames)) {
      throw new Error('predict() must take an object where the keys are the named inputs of the model.')
    }
    if (!every(inputNames, inputName => inputData[inputName] instanceof Float32Array)) {
      throw new Error('predict() must take an object where the values are the flattened data as Float32Array.')
    }

    inputNames.forEach(inputName => {
      this.inputTensors[inputName].tensor.data = inputData[inputName]
    })
  }
}
