<template>
  <div class="demo">
    <imagenet
      modelName="resnet50"
      :modelFilepath="modelFilepath"
      :hasWebGL="hasWebGL"
      :imageSize="224"
      :visualizations="['None']"
      :drawArchitectureDiagramPaths="drawArchitectureDiagramPaths"
    ></imagenet>
    <div v-resize="drawArchitectureDiagramPaths" class="architecture-container">
      <div v-for="(row, rowIndex) in architectureDiagramRows" :key="`row-${rowIndex}`" class="layers-row">
        <div v-for="layer in row" :key="`layer-${layer.name}`" class="layer-column">
          <div
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
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/resnet50-arch'
import Imagenet from '../common/Imagenet'

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/resnet50/resnet50.bin'
const MODEL_FILEPATH_DEV = '/demos/data/resnet50/resnet50.bin'

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
      for (let row = 1; row < 168; row++) {
        let cols = []
        for (let col = 0; col < 2; col++) {
          cols.push(_.find(this.architectureDiagram, { row, col }) || {})
        }
        rows.push(cols)
      }
      return rows
    }
  },

  methods: {
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
  min-width: 700px;
  max-width: 900px;
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
