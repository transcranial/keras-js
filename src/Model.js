/* global XMLHttpRequest */
import Promise from 'bluebird'
import toPairs from 'lodash/toPairs'
import mapKeys from 'lodash/mapKeys'
import camelCase from 'lodash/camelCase'
import find from 'lodash/find'
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
    if (filepaths.model && filepaths.weights && filepaths.metadata) {
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

    // map of model layers
    this.modelLayers = new Map()

    // directed acyclic graph of model network
    this.modelDAG = {}

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
  async initialize () {
    const dataTypes = ['model', 'weights', 'metdata']
    try {
      // get data
      await Promise.all(dataTypes.map(type => {
        return this.dataRequest(type, this.config.headers)
      }))
      // create layers
      this.createLayers()
      return Promise.resolve()
    } catch (err) {
      console.log(err)
      this.interrupt()
      return Promise.reject(err)
    }
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
        resolve()
      }
      xhr.onerror = e => reject(e)
      xhr.send(null)
      this.xhrs[type] = xhr
    })
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

      modelConfig.forEach((layerConfig, index) => {
        // create Input node at start
        if (index === 0) {
          const layer = new Input({
            name: inputName,
            inputShape: layerConfig.batch_input_shape.slice(1)
          })
          this.modelLayers.set(inputName, layer)
          this.modelDAG[inputName] = {
            name: inputName,
            outbound: []
          }
        }

        const layerClass = layerConfig.class_name
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

          this.modelLayers.set(layerConfig.name, layer)
          this.modelDAG[layerConfig.name] = {
            name: layerConfig.name,
            outbound: []
          }
          if (index === 0) {
            this.modelDAG[inputName].outbound.push(layerConfig.name)
          } else {
            this.modelDAG[modelConfig[index - 1].name].outbound.push(layerConfig.name)
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
  predict (data) {
    
  }
}
