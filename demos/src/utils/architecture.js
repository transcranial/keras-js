import _ from 'lodash'

/**
 * Creates data for architecture diagram component
 *
 * @param {Object[]} modelLayersInfo
 */
export function createDiagramData(modelLayersInfo) {
  const diagramLayers = {}
  const diagramConnections = []

  // recursive function to traverse directed acyclic graph
  const traverse = nodes => {
    if (nodes.length === 0) return

    nodes.forEach((node, i) => {
      if (_.has(diagramLayers, node)) return

      const layerInfo = _.find(modelLayersInfo, ['name', node])
      const maxRow = _.max(layerInfo.inbound.map(nodeIn => _.get(diagramLayers, [nodeIn, 'row'], 0))) || 0
      const maxCol = _.max(layerInfo.inbound.map(nodeIn => _.get(diagramLayers, [nodeIn, 'col'], 0))) || 0

      let row = maxRow + 1
      let col = maxCol + i
      let existing = _.find(diagramLayers, { row, col })
      while (existing) {
        col += 1
        existing = _.find(diagramLayers, { row, col })
      }

      const diagramLayer = { ...layerInfo, row, col }
      diagramLayers[node] = diagramLayer

      layerInfo.inbound.forEach(nodeIn => {
        diagramConnections.push({ from: nodeIn, to: node })
      })

      traverse(layerInfo.outbound)
    })
  }

  const inputLayerNames = _.map(_.filter(modelLayersInfo, layerInfo => !layerInfo.inbound.length), 'name')
  traverse(inputLayerNames)

  // shift inbound nodes from same column into new columns
  _.keys(diagramLayers).forEach(node => {
    const sameColNodes = diagramLayers[node].inbound.filter(nodeIn => {
      return diagramLayers[nodeIn].col === diagramLayers[node].col
    })
    if (sameColNodes.length > 1) {
      diagramLayers[node].col += 1
    }
  })

  // add corners to diagram connections
  diagramConnections.forEach(connection => {
    const { from, to } = connection
    const fromRow = diagramLayers[from].row
    const fromCol = diagramLayers[from].col
    const toRow = diagramLayers[to].row
    const toCol = diagramLayers[to].col
    if (fromRow < toRow && fromCol < toCol) {
      if (!_.find(diagramLayers, o => o.row > fromRow && o.row <= toRow && o.col === fromCol)) {
        connection.corner = 'bottom-left'
      } else if (!_.find(diagramLayers, o => o.row >= fromRow && o.row < toRow && o.col === toCol)) {
        connection.corner = 'top-right'
      }
    } else if (fromRow < toRow && fromCol > toCol) {
      if (!_.find(diagramLayers, o => o.row > fromRow && o.row <= toRow && o.col === fromCol)) {
        connection.corner = 'bottom-right'
      } else if (!_.find(diagramLayers, o => o.row >= fromRow && o.row < toRow && o.col === toCol)) {
        connection.corner = 'top-left'
      }
    }
  })

  const diagramRows = []
  const maxRow = _.max(_.map(diagramLayers, 'row'))
  const maxCol = _.max(_.map(diagramLayers, 'col'))
  for (let row = 0; row <= maxRow; row++) {
    let diagramCols = []
    for (let col = 0; col <= maxCol; col++) {
      diagramCols.push(_.find(diagramLayers, { row, col }) || {})
    }
    diagramRows.push(diagramCols)
  }

  return { diagramRows, diagramConnections }
}
