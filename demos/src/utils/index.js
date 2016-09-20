/* global ImageData */
import sum from 'lodash/sum'
import flatten from 'lodash/flatten'
import unpack from 'ndarray-unpack'

/**
 * Find mindpoint of two points
 */
export function getMidpoint (p1, p2) {
  const [x1, y1] = p1
  const [x2, y2] = p2
  return [
    x1 + (x2 - x1) / 2,
    y1 + (y2 - y1) / 2
  ]
}

/**
 * Gets the (x, y) coordinates of an UI event relative to its target,
 * e.g., canvas. Accounts for touch events as well as mouse events.
 */
export function getCoordinates (e) {
  let { clientX, clientY } = e
  // for touch event
  if (e.touches && e.touches.length) {
    clientX = e.touches[0].clientX
    clientY = e.touches[0].clientY
  }
  const { left, top } = e.target.getBoundingClientRect()
  const [x, y] = [clientX - left, clientY - top]
  return [x, y]
}

/**
 * Centers and crops canvas ImageData based on alpha channel.
 * @param {ImageData} imageData
 * @returns {ImageData}
 */
export function centerCrop (imageData) {
  const { data, width, height } = imageData
  let [xmin, ymin] = [width, height]
  let [xmax, ymax] = [-1, -1]
  for (let i = 0; i < width; i++) {
    for (let j = 0; j < height; j++) {
      const idx = (i + j * width)
      if (data[4 * idx + 3] > 0) {
        if (i < xmin) xmin = i
        if (i > xmax) xmax = i
        if (j < ymin) ymin = j
        if (j > ymax) ymax = j
      }
    }
  }

  // add a little padding
  xmin -= 20
  xmax += 20
  ymin -= 20
  ymax += 20

  // make bounding box square
  let [widthNew, heightNew] = [(xmax - xmin + 1), (ymax - ymin + 1)]
  if (widthNew < heightNew) {
    // new width < new height
    const halfBefore = Math.floor((heightNew - widthNew) / 2)
    const halfAfter = heightNew - widthNew - halfBefore
    xmax += halfAfter
    xmin -= halfBefore
  } else if (widthNew > heightNew) {
    // new width > new height
    const halfBefore = Math.floor((widthNew - heightNew) / 2)
    const halfAfter = widthNew - heightNew - halfBefore
    ymax += halfAfter
    ymin -= halfBefore
  }

  widthNew = xmax - xmin + 1
  heightNew = ymax - ymin + 1
  let dataNew = new Uint8ClampedArray(widthNew * heightNew * 4)
  for (let i = xmin; i <= xmax; i++) {
    for (let j = ymin; j <= ymax; j++) {
      if (i >= 0 && i < width && j >= 0 && j < height) {
        const idx = (i + j * width)
        const idxNew = ((i - xmin) + (j - ymin) * widthNew)
        dataNew[4 * idxNew + 3] = data[4 * idx + 3]
      }
    }
  }

  return new ImageData(dataNew, widthNew, heightNew)
}

/**
 * calculates mean and stddev for a ndarray tensor
 */
export function tensorStats (tensor) {
  const mean = sum(tensor.data) / tensor.data.length
  const stddev = Math.sqrt(sum(tensor.data.map(x => (x - mean) ** 2)) / tensor.data.length)
  return { mean, stddev }
}

/**
 * calculates min and max for a ndarray tensor
 */
export function tensorMinMax (tensor) {
  const min = Math.min(...tensor.data)
  const max = Math.max(...tensor.data)
  return { min, max }
}

/**
 * Takes in a ndarray of shape [x]
 * and creates image data
 */
export function image1Dtensor (tensor) {
  const { min, max } = tensorMinMax(tensor)
  let imageData = new Uint8ClampedArray(tensor.size * 4)
  for (let i = 0, len = imageData.length; i < len; i += 4) {
    imageData[i + 3] = 255 * (tensor.data[i / 4] - min) / (max - min)
  }
  return new ImageData(imageData, tensor.shape[0], 1)
}

/**
 * Takes in a ndarray of shape [x, y]
 * and creates image data
 */
export function image2Dtensor (tensor) {
  const { min, max } = tensorMinMax(tensor)
  let imageData = new Uint8ClampedArray(tensor.size * 4)
  for (let i = 0, len = imageData.length; i < len; i += 4) {
    imageData[i + 3] = 255 * (tensor.data[i / 4] - min) / (max - min)
  }
  return new ImageData(imageData, tensor.shape[0], tensor.shape[1])
}

/**
 * Takes in a ndarray of shape [x, y, z]
 * and creates an array of z ImageData [x, y] elements
 */
export function unroll3Dtensor (tensor) {
  const { min, max } = tensorMinMax(tensor)
  let shape = tensor.shape.slice()
  let unrolled = []
  for (let k = 0, channels = shape[2]; k < channels; k++) {
    const channelData = flatten(unpack(tensor.pick(null, null, k)))
    unrolled.push(channelData)
  }

  return unrolled.map(channelData => {
    let imageData = new Uint8ClampedArray(channelData.length * 4)
    for (let i = 0, len = channelData.length; i < len; i++) {
      imageData[i * 4] = 0
      imageData[i * 4 + 1] = 0
      imageData[i * 4 + 2] = 0
      imageData[i * 4 + 3] = 255 * (channelData[i] - min) / (max - min)
    }
    return new ImageData(imageData, shape[0], shape[1])
  })
}
