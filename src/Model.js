import Promise from 'bluebird'
import axios from 'axios'
import toPairs from 'lodash/toPairs'
import mapKeys from 'lodash/mapKeys'
import find from 'lodash/find'
import keys from 'lodash/keys'
import values from 'lodash/values'
import sum from 'lodash/sum'
import isEqual from 'lodash/isEqual'
import every from 'lodash/every'
import * as layers from './layers'
import Tensor from './Tensor'

const axiosSource = axios.CancelToken.source()

/**
 * Model class
 */
export default class Model {
  /**
   * create new Model class
   * @param {object} config.filepaths
   * @param {string} config.filepaths.modelFilepath - path to model architecture configuration (json)
   * @param {string} config.filepaths.weightsFilepath - path to weights data (arraybuffer)
   * @param {string} config.filepaths.metadataFilepath - path to weights metadata (json)
   * @param {object} [config.headers] - any additional HTTP headers required for resource fetching
   * @param {boolean} [config.gpu] - enable GPU
   * @param {boolean} [config.pipeline] - configure capable layers to run in pipeline mode (gpu must be enabled)
   * @param {boolean} [config.layerCallPauses] - force next tick after each layer call
   */
  constructor(config = {}) {
    const {
      filepaths = {},
      headers = {},
      filesystem = false,
      gpu = false,
      pipeline = false,
      layerCallPauses = false
    } = config

    if (!filepaths.model || !filepaths.weights || !filepaths.metadata) {
      throw new Error('File paths must be declared for model, weights, and metadata.')
    }
    this.filepaths = filepaths
    this.filetypes = { model: 'json', weights: 'arraybuffer', metadata: 'json' }

    // HTTP(S) headers used during data fetching
    this.headers = headers

    // specifies that data files are from local file system
    // only in node
    this.filesystem = typeof window !== 'undefined' ? false : filesystem

    // flag to enable GPU where possible (disable in node environment)
    this.gpu = typeof window !== 'undefined' ? gpu : false
    // flag to enable GPU pipeline mode where possible
    this.pipeline = this.gpu ? pipeline : false
    // flag to enable 0 ms pauses after layer computation calls
    this.layerCallPauses = layerCallPauses

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

    // data request progress
    this.dataRequestProgress = { model: 0, weights: 0, metadata: 0 }

    // map of model layers
    this.modelLayersMap = new Map()

    // array of model layer names with result
    this.layersWithResults = []

    // directed acyclic graph of model network
    this.modelDAG = {}

    // input tensors
    this.inputTensors = {}

    // Promise for when Model class is initialized
    this._ready = this._initialize()

    // flag while computations are being performed
    this.isRunning = false
  }

  /**
   * Promise for when model data is loaded and layers are initialized.
   * @returns {Promise}
   */
  ready() {
    return this._ready
  }

  /**
   * Cancels any existing data requests
   */
  _interrupt() {
    axiosSource.cancel()
  }

  /**
   * Model initialization
   * @returns {Promise}
   */
  _initialize() {
    const dataTypes = ['model', 'weights', 'metadata']
    return Promise.all(
      dataTypes.map(type => {
        return this.filesystem ? this._dataRequestFS(type) : this._dataRequestHTTP(type, this.headers)
      })
    )
      .then(() => {
        this._createLayers()
        return Promise.resolve()
      })
      .catch(err => {
        console.log(err)
        this._interrupt()
      })
  }

  /**
   * Makes data FS request (node only)
   * @async
   * @param {string} type - type of requested data, one of `model`, `weights`, or `metadata`.
   * @returns {Promise}
   */
  _dataRequestFS(type) {
    const readFile = Promise.promisify(require('fs').readFile)
    const filetype = this.filetypes[type]
    const encoding = filetype === 'json' ? 'utf8' : undefined
    return readFile(this.filepaths[type], encoding)
      .then(data => {
        if (filetype === 'json') {
          this.data[type] = JSON.parse(data)
        } else if (filetype === 'arraybuffer') {
          this.data[type] = data.buffer
        } else {
          throw new Error(`Invalid file type: ${filetype}`)
        }
        this.dataRequestProgress[type] = 100
      })
      .catch(err => {
        throw err
      })
  }

  /**
   * Makes data HTTP request (browser or node)
   * @async
   * @param {string} type - type of requested data, one of `model`, `weights`, or `metadata`.
   * @param {Object} [headers] - any headers to be passed along with request
   * @returns {Promise}
   */
  _dataRequestHTTP(type, headers = {}) {
    return axios
      .get(this.filepaths[type], {
        responseType: this.filetypes[type],
        headers,
        onDownloadProgress: e => {
          if (e.lengthComputable) {
            const percentComplete = Math.round(100 * e.loaded / e.total)
            this.dataRequestProgress[type] = percentComplete
          }
        },
        cancelToken: axiosSource.token
      })
      .then(res => {
        this.data[type] = res.data
        this.dataRequestProgress[type] = 100
      })
      .catch(err => {
        if (axios.isCancel(err)) {
          console.log('Data request canceled', err.message)
        } else {
          throw err
        }
      })
  }

  /**
   * Loading progress calculated from all the data requests combined.
   * @returns {number} progress
   */
  getLoadingProgress() {
    const progressValues = values(this.dataRequestProgress)
    return Math.round(sum(progressValues) / progressValues.length)
  }

  /**
   * Toggle GPU mode on/off
   * Iterate through all layers and set `gpu` attribute
   * @param {boolean} mode - on/off
   */
  toggleGpu(mode) {
    if (typeof mode === 'undefined') {
      this.gpu = !this.gpu
    } else {
      this.gpu = mode
    }
    for (let layer of this.modelLayersMap.values()) {
      layer.toggleGpu(this.gpu)
    }
  }

  /**
   * Builds network layer DAG
   *
   * For Keras models of class Sequential, we still convert the list into DAG format
   * for straightforward interoperability with graph models. We must first create an
   * Input layer as the initial layer, however.
   *
   * For class Model, the network DAG is constructed from the configuration inbound
   * and outbound nodes. Note that Models can have layers be entire Sequential branches.
   */
  _createLayers() {
    const modelClass = this.data.model.class_name

    let modelConfig = []
    if (modelClass === 'Sequential') {
      modelConfig = this.data.model.config
    } else if (modelClass === 'Model') {
      modelConfig = this.data.model.config.layers
    }

    if (!(Array.isArray(modelConfig) && modelConfig.length)) {
      throw new Error('Model configuration does not contain any layers.')
    }

    modelConfig.forEach((layerDef, index) => {
      const layerClass = layerDef.class_name
      const layerConfig = layerDef.config

      if (modelClass === 'Model' && layerClass === 'Sequential') {
        // when layer is a Sequential branch in a Model
        layerConfig.forEach((branchLayerDef, branchIndex) => {
          const branchLayerClass = branchLayerDef.class_name
          const branchLayerConfig = branchLayerDef.config

          const branchInboundLayerNames = branchIndex === 0
            ? layerDef.inbound_nodes[0].map(node => node[0])
            : [layerConfig[branchIndex - 1].config.name]

          this._createLayer(branchLayerClass, branchLayerConfig, branchInboundLayerNames)
        })
      } else if (!(layerClass in layers)) {
        throw new Error(`Layer ${layerClass} specified in model configuration is not implemented!`)
      } else {
        // create InputLayer node for Sequential class (which is not explicitly defined in config)
        // create input tensor for InputLayer specified in Model class (layer itself created later)
        if (modelClass === 'Sequential' && index === 0) {
          const inputName = 'input'
          const inputShape = layerConfig.batch_input_shape.slice(1)
          const layer = new layers.InputLayer({ name: inputName, shape: inputShape })
          this.modelLayersMap.set(inputName, layer)
          this.modelDAG[inputName] = { layerClass: 'InputLayer', name: inputName, inbound: [], outbound: [] }
          this.inputTensors[inputName] = new Tensor([], inputShape)
        } else if (modelClass === 'Model' && layerClass === 'InputLayer') {
          const inputShape = layerConfig.batch_input_shape.slice(1)
          this.inputTensors[layerConfig.name] = new Tensor([], inputShape)
        }

        let inboundLayerNames = []
        if (modelClass === 'Sequential') {
          if (index === 0) {
            inboundLayerNames = ['input']
          } else {
            inboundLayerNames = [modelConfig[index - 1].config.name]
          }
        } else if (modelClass === 'Model') {
          const inboundNodes = layerDef.inbound_nodes
          if (inboundNodes && inboundNodes.length) {
            inboundLayerNames = inboundNodes[0].map(node => node[0])
          }
        }

        this._createLayer(layerClass, layerConfig, inboundLayerNames)
      }
    })
  }

  /**
   * Create single layer.
   * @param {String} layerClass
   * @param {Object} layerConfig
   * @param {Array<String>} inboundLayerNames
   * @param {Boolean} isSequential
   */
  _createLayer(layerClass, layerConfig, inboundLayerNames) {
    let layer
    if (layerClass === 'Bidirectional' || layerClass === 'TimeDistributed') {
      // create wrapper layers
      const wrappedLayerConfig = layerConfig.layer.config
      const wrappedLayerClass = layerConfig.layer.class_name
      wrappedLayerConfig.gpu = this.gpu

      layer = new layers[layerClass](
        Object.assign({}, layerConfig, { layer: new layers[wrappedLayerClass](wrappedLayerConfig) })
      )
    } else {
      // create regular layers
      layer = new layers[layerClass](Object.assign({ gpu: this.gpu, pipeline: this.pipeline }, layerConfig))
    }

    // layer weights
    let weightNames = []
    if (layerClass === 'Bidirectional') {
      const forwardWeightNames = layer.forwardLayer.params.map(
        param => `${layerConfig.name}/forward_${layerConfig.layer.config.name}/${param}`
      )
      const backwardWeightNames = layer.backwardLayer.params.map(
        param => `${layerConfig.name}/backward_${layerConfig.layer.config.name}/${param}`
      )
      weightNames = forwardWeightNames.concat(backwardWeightNames)
    } else if (layerClass === 'TimeDistributed') {
      weightNames = layer.layer.params.map(param => `${layerConfig.name}/${param}`)
    } else {
      weightNames = layer.params.map(param => `${layerConfig.name}/${param}`)
    }
    if (weightNames && weightNames.length) {
      const weights = weightNames.map(weightName => {
        const paramMetadata = find(this.data.metadata, meta => {
          const weightRE = new RegExp(`^${weightName}`)
          return weightRE.test(meta.weight_name)
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
    this.modelDAG[layerConfig.name] = { layerClass, name: layerConfig.name, inbound: [], outbound: [] }

    inboundLayerNames.forEach(layerName => {
      this.modelDAG[layerConfig.name].inbound.push(layerName)
      this.modelDAG[layerName].outbound.push(layerConfig.name)
    })
  }

  /**
   * Runs .call() on merge layer
   * @param {Layer} currentLayer
   * @param {Layer[]} inboundLayers
   * @param {boolean} copyBeforeCall
   * @returns {Tensor}
   */
  _mergeLayerCall(currentLayer, inboundLayers, copyBeforeCall) {
    let inputs = inboundLayers.map(layer => layer.result)
    const canRunInPipeline = inputs.every(x => x._fromPipeline)
    if (!canRunInPipeline || !currentLayer._pipelineEnabled) {
      // If currentLayer is not pipeline enabled, then all inbound results
      // must first be converted from weblas tensors to regular tensors, if
      // necessary.
      // If currentLayer is pipeline enabled, but not all inbound results are
      // from pipeline mode, then all must still be converted from weblas
      // tensors to regular tensors.
      inputs = inputs.map((x, i) => {
        if (x._fromPipeline) {
          // copy from weblas tensor into regular tensor
          return inboundLayers[i].transferFromPipeline(x)
        } else if (copyBeforeCall) {
          // make a copy of regular tensor
          return new Tensor(x.tensor.data, x.tensor.shape)
        }
        return x
      })
    } else if (copyBeforeCall) {
      // If currentLayer is pipeline enabled, and all inbound results are from
      // pipeline mode as well, but there are sibling layer nodes that require
      // the same input(s) (thus copyBeforeCall is true), then we directly copy
      // the weblas tensors.
      inputs = inputs.map(x => {
        let xNew = new Tensor([], x.tensor.shape)
        xNew.copyFromWeblasTensor(x.weblasTensor)
        xNew._fromPipeline = true
        xNew._actualShape = x._actualShape.slice()
        return xNew
      })
    }

    return currentLayer.call(inputs)
  }

  /**
   * Runs .call() on regular layer
   * @param {Layer} currentLayer
   * @param {Layer} inboundLayer
   * @param {boolean} copyBeforeCall
   * @returns {Tensor}
   */
  _regularLayerCall(currentLayer, inboundLayer, copyBeforeCall) {
    let inboundLayerResult = inboundLayer.result
    if (!inboundLayerResult._fromPipeline || !currentLayer._pipelineEnabled) {
      // If currentLayer is not pipeline enabled or inbound layer result is not
      // from pipeline mode, then result must first be converted from a weblas
      // tensor to a regular tensor, if necessary.
      if (inboundLayerResult._fromPipeline) {
        // copy from weblas tensor into regular tensor
        inboundLayerResult = inboundLayer.transferFromPipeline(inboundLayerResult)
      } else if (copyBeforeCall) {
        // make a copy of regular tensor
        inboundLayerResult = new Tensor(inboundLayerResult.tensor.data, inboundLayerResult.tensor.shape)
      }
    } else if (copyBeforeCall) {
      // If currentLayer is pipeline enabled, and prev layer result is from
      // pipeline mode as well, but there are sibling layer nodes that require
      // the same input (thus copyBeforeCall is true), then we directly copy
      // the weblas tensor.
      let xNew = new Tensor([], inboundLayerResult.tensor.shape)
      xNew.copyFromWeblasTensor(inboundLayerResult.weblasTensor)
      xNew._fromPipeline = true
      xNew._actualShape = inboundLayerResult._actualShape.slice()
      inboundLayerResult = xNew
    }

    return currentLayer.call(inboundLayerResult)
  }

  /**
   * Async function for recursively traversing the DAG
   * Graph object is stored in `this.modelDAG`, keyed by layer name.
   * Layers are retrieved from Map object `this.modelLayersMap`.
   * @async
   * @param {[]string} nodes - array of layer names
   * @returns {Promise.<boolean>}
   */
  async _traverseDAG(nodes) {
    if (nodes.length === 0) {
      // Stopping criterion:
      // an output node will have 0 outbound nodes.
      return true
    } else if (nodes.length === 1) {
      // Where computational logic lives for a given layer node
      // - Makes sure results are available from inbound layer nodes
      // - Keeps generator going until results are available from inbound layer nodes
      //   (important for merge layer nodes where multiple inbound nodes may
      //    complete asynchronously)
      // - Runs computation for current layer node: .call()
      // - Starts new generator function for outbound nodes
      const node = nodes[0]
      const { layerClass, inbound, outbound } = this.modelDAG[node]
      if (layerClass !== 'InputLayer') {
        let currentLayer = this.modelLayersMap.get(node)
        if (currentLayer.visited) {
          return false
        }

        const inboundLayers = inbound.map(n => this.modelLayersMap.get(n))
        if (!every(inboundLayers.map(layer => layer.hasResult))) {
          return false
        }

        const numSiblingNodes = inbound
          .map(n => this.modelDAG[n].outbound)
          .reduce((num, outbound) => num + outbound.length, 0)
        const copyBeforeCall = numSiblingNodes >= 1

        if (['Merge', 'Add', 'Multiply', 'Average', 'Maximum', 'Concatenate', 'Dot'].includes(layerClass)) {
          currentLayer.result = this._mergeLayerCall(currentLayer, inboundLayers, copyBeforeCall)
        } else {
          currentLayer.result = this._regularLayerCall(currentLayer, inboundLayers[0], copyBeforeCall)
        }

        currentLayer.hasResult = true
        currentLayer.visited = true
        this.layersWithResults.push(currentLayer.name)
        if (this.layerCallPauses) {
          // temporarily pause 0 ms
          // useful for allowing DOM operations and other simultaneously running functions on the main thread
          await Promise.delay(0)
        }
      } else {
        this.layersWithResults.push(this.modelLayersMap.get(node).name)
      }
      await this._traverseDAG(outbound)
    } else {
      await Promise.all(nodes.map(node => this._traverseDAG([node])))
    }
  }

  /**
   * Predict
   * @async
   * @param {Object} inputData - object where the keys are the named inputs of the model,
   *                             and values the TypedArray numeric data
   * @returns {Promise.<Object>} - outputData object where the keys are the named outputs
   *                             of the model, and values the TypedArray numeric data
   */
  async predict(inputData) {
    this.isRunning = true

    const inputNames = keys(this.inputTensors).sort()
    if (!isEqual(keys(inputData).sort(), inputNames)) {
      this.isRunning = false
      throw new Error(`predict() must take an object where the keys are the named inputs of the model: ${inputNames}.`)
    }
    if (!every(inputNames, inputName => inputData[inputName] instanceof Float32Array)) {
      this.isRunning = false
      throw new Error('predict() must take an object where the values are the flattened data as Float32Array.')
    }

    // reset hasResult and visited flags in all layers
    this.layersWithResults = []
    for (let layer of this.modelLayersMap.values()) {
      layer.hasResult = false
      layer.visited = false
    }

    // load data to input tensors
    inputNames.forEach(inputName => {
      let inputLayer = this.modelLayersMap.get(inputName)
      this.inputTensors[inputName].replaceTensorData(inputData[inputName])
      inputLayer.result = inputLayer.call(this.inputTensors[inputName])
      inputLayer.hasResult = true
      inputLayer.visited = true
    })

    // start traversing DAG at input
    await this._traverseDAG(inputNames)

    // outputs are layers with no outbound nodes
    const modelClass = this.data.model.class_name
    if (modelClass === 'Sequential') {
      const outputLayer = find(values(this.modelDAG), node => !node.outbound.length)
      const { result } = this.modelLayersMap.get(outputLayer.name)
      const outputData = { output: result.tensor.data }
      this.isRunning = false
      return outputData
    } else if (modelClass === 'Model') {
      const outputLayers = values(this.modelDAG).filter(node => !node.outbound.length)
      let outputData = {}
      outputLayers.forEach(layer => {
        const { result } = this.modelLayersMap.get(layer.name)
        outputData[layer.name] = result.tensor.data
      })
      this.isRunning = false
      return outputData
    }
  }
}
