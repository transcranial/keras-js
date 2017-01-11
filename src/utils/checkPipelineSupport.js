export default function checkPipelineSupport (layerClass, attrs) {
  return false
  switch (layerClass) {
    case 'Convolution2D':
      if (
        attrs.activation === 'linear' ||
        attrs.activation === 'relu'
      ) {
        return true
      }
      return false

    case 'BatchNormalization':
      if (
        attrs.mode === 0
      ) {
        return true
      }
      return false

    case 'MaxPooling2D':
    case 'AveragePooling2D':
      return true

    case 'Merge':
      if (
        attrs.mode === 'concat'
      ) {
        return true
      }
      return false

    default:
      return false
  }
}
