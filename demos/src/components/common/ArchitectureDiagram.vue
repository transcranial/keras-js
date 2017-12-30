<template>
  <div v-resize="drawDiagramPaths" class="architecture-container">
    <div v-for="(row, rowIndex) in diagramRows" :key="`row:${rowIndex}`" class="layers-row">
      <div v-for="(layer, colIndex) in row" :key="`layer:${rowIndex},${colIndex}`" class="layer-column">
        <div v-if="layer.layerClass" class="layer" :id="layer.name">
          <div class="layer-class-name">{{ layer.layerClass }}</div>
          <div class="layer-description"> {{ layer.description }}</div>
        </div>
      </div>
    </div>
    <svg class="architecture-connections" width="100%" height="100%">
      <g>
        <path v-for="(path, pathIndex) in diagramPaths" :key="`path-${pathIndex}`" :d="path" />
      </g>
    </svg>
  </div>
</template>

<script>
import _ from 'lodash'
import { architectureUtils } from '../../utils'

export default {
  props: ['modelLayersInfo'],

  data() {
    return {
      diagramRows: [],
      diagramConnections: [],
      diagramPaths: []
    }
  },

  watch: {
    modelLayersInfo(newVal) {
      const { diagramRows, diagramConnections } = architectureUtils.createDiagramData(newVal)
      this.diagramRows = diagramRows
      this.diagramConnections = diagramConnections
      this.drawDiagramPaths()
    }
  },

  methods: {
    drawDiagramPaths: _.debounce(function() {
      this.diagramPaths = []
      this.diagramConnections.forEach(conn => {
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
          path = `M${xFrom},${yFrom} L${xFrom},${yTo - 10} Q${xFrom},${yTo} ${xFrom - 10},${yTo} L${xTo},${yTo}`
        }

        this.diagramPaths.push(path)
      })
    }, 100)
  }
}
</script>

<style lang="postcss" scoped>
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
        width: 150px;
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

        & .layer-description {
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
      stroke-width: 3px;
      stroke: #aaaaaa;
      fill: none;
    }
  }
}
</style>
