export default function checkPipelineSupport(layerClass, attrs) {
  switch (layerClass) {
    case 'Activation':
      if (['linear', 'relu'].indexOf(attrs.activation) > -1) {
        return true
      }
      return false

    case 'Conv2D':
      if (['linear', 'relu'].indexOf(attrs.activation) > -1) {
        return true
      }
      return false

    case 'BatchNormalization':
      if (attrs.mode === 0) {
        return true
      }
      return false

    case 'MaxPooling2D':
    case 'AveragePooling2D':
      return true

    case 'Merge':
      if (['concat', 'sum', 'mul', 'ave', 'max'].indexOf(attrs.mode) > -1) {
        return true
      }
      return false

    default:
      return false
  }
}
