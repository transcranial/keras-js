<template>
  <div class="demo">
    <imagenet
      modelName="densenet121"
      :modelFilepath="modelFilepath"
      :hasWebGL="hasWebGL"
      :imageSize="224"
      :visualizations="['CAM']"
      :preprocess="preprocess"
      :drawArchitectureDiagramPaths="drawArchitectureDiagramPaths"
    ></imagenet>
    <!-- <div v-resize="drawArchitectureDiagramPaths" class="architecture-container">
      <div v-for="(row, rowIndex) in architectureDiagramRows" :key="`row-${rowIndex}`" class="layers-row">
        <div v-for="(layers, i) in row" :key="`layer-row-${i}`" class="layer-column">
          <div v-for="layer in layers" :key="`layer-${layer.name}`"
            v-if="layer.className"
            class="layer"
            :id="layer.name"
          >
            <div class="layer-class-name">{{ layer.className }}</div>
            <div class="layer-details"> {{ layer.details }}</div>
          </div>
        </div>
      </div>
      <svg class="architecture-connections" width="100%" height="100%">
        <g>
          <path v-for="(path, pathIndex) in architectureDiagramPaths" :key="`path-${pathIndex}`" :d="path" />
        </g>
      </svg>
    </div> -->
  </div>
</template>

<script>
import _ from 'lodash'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/densenet121-arch'
import Imagenet from '../common/Imagenet'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/densenet121/densenet121.bin'
const MODEL_FILEPATH_DEV = '/demos/data/densenet121/densenet121.bin'

export default {
  props: ['hasWebGL'],

  components: { Imagenet },

  data() {
    return {
      modelFilepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: []
    }
  },

  computed: {
    architectureDiagramRows() {
      let rows = []
      for (let row = 0; row < 112; row++) {
        let cols = []
        for (let col = 0; col < 4; col++) {
          cols.push(_.filter(this.architectureDiagram, { row, col }))
        }
        rows.push(cols)
      }
      return rows
    }
  },

  methods: {
    preprocess(imageData) {
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
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
    },
    drawArchitectureDiagramPaths: _.debounce(function() {
      this.architectureDiagramPaths = []
      this.architectureConnections.forEach(conn => {
        const containerElem = document.getElementsByClassName('architecture-container')[0]
        const fromElem = document.getElementById(conn.from)
        const toElem = document.getElementById(conn.to)
        const containerElemCoords = containerElem.getBoundingClientRect()
        const fromElemCoords = fromElem.getBoundingClientRect()
        const toElemCoords = toElem.getBoundingClientRect()
        const xContainer = containerElemCoords.left
        const yContainer = containerElemCoords.top
        const xFrom = fromElemCoords.left + fromElemCoords.width / 2 - xContainer
        const yFrom = fromElemCoords.top + fromElemCoords.height / 2 - yContainer
        const xTo = toElemCoords.left + toElemCoords.width / 2 - xContainer
        const yTo = toElemCoords.top + toElemCoords.height / 2 - yContainer

        let path = `M${xFrom},${yFrom} L${xTo},${yTo}`
        if (conn.corner === 'top-right') {
          path = `M${xFrom},${yFrom} L${xTo - 10},${yFrom} Q${xTo},${yFrom} ${xTo},${yFrom + 10} L${xTo},${yTo}`
        } else if (conn.corner === 'bottom-left') {
          path = `M${xFrom},${yFrom} L${xFrom},${yTo - 10} Q${xFrom},${yTo} ${xFrom + 10},${yTo} L${xTo},${yTo}`
        } else if (conn.corner === 'top-left') {
          path = `M${xFrom},${yFrom} L${xTo + 10},${yFrom} Q${xTo},${yFrom} ${xTo},${yFrom + 10} L${xTo},${yTo}`
        } else if (conn.corner === 'bottom-right') {
          path = `M${xFrom},${yFrom} L${xFrom},${yFrom + 20} Q${xFrom},${yFrom + 30} ${xFrom - 10},${yFrom +
            30} L${xTo + 10},${yFrom + 30} Q${xTo},${yFrom + 30} ${xTo},${yFrom + 40} L${xTo},${yTo}`
        }

        this.architectureDiagramPaths.push(path)
      })
    }, 100)
  }
}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.architecture-container {
  min-width: 800px;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;

  & .layers-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    margin-bottom: 5px;
    position: relative;
    z-index: 1;

    & .layer-column {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 5px;

      & .layer {
        display: inline-block;
        background: whitesmoke;
        border: 2px solid var(--color-green);
        border-radius: 5px;
        padding: 2px 5px 0px;
        margin: 3px;

        & .layer-class-name {
          color: var(--color-green);
          font-size: 12px;
          font-weight: bold;
        }

        & .layer-details {
          color: #999999;
          font-size: 11px;
          font-weight: bold;
        }
      }
    }
  }

  & .architecture-connections {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 0;

    & path {
      stroke-width: 4px;
      stroke: #aaaaaa;
      fill: none;
    }
  }
}
</style>
