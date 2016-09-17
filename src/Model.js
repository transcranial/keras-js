/* global XMLHttpRequest */
import Promise from 'bluebird'
import toPairs from 'lodash/toPairs'
import mapKeys from 'lodash/mapKeys'
import camelCase from 'lodash/camelCase'
import * as layers from '../layers'

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

      modelConfig.forEach((layerConfig, index) => {
        const layerClass = layerConfig.class_name
        if (layerClass in layers) {
          const attrs = mapKeys(layerConfig, (v, k) => camelCase(k))
          const layer = new layers[layerClass](attrs)
          this.modelLayers.set(attrs.name, layer)
          this.modelDAG[
        } else {
          throw new Error(`Layer ${layerClass} specified in model configuration is not implemented!`)
        }
      })
    }
  }
}
