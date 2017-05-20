export const ARCHITECTURE_DIAGRAM = [
  { name: 'input_2', className: 'InputLayer', details: '100-dimensional noise vector', row: 0 },
  { name: 'input_3', className: 'InputLayer', details: 'digit', row: 0 },
  { name: 'embedding_1', className: 'Embedding', details: '100-dimensional vector', row: 1 },
  { name: 'flatten_2', className: 'Flatten', details: '', row: 2 },
  { name: 'multiply_1', className: 'Multiply', details: 'merge', row: 3 },
  { name: 'dense_1', className: 'Dense', details: 'ReLU activation, output dimensions = 1024', row: 4 },
  { name: 'dense_2', className: 'Dense', details: 'ReLU activation, output dimensions = 6272 (128 x 7 x 7)', row: 5 },
  { name: 'reshape_1', className: 'Reshape', details: '6272 -> (7, 7, 128)', row: 6 },
  { name: 'up_sampling2d_1', className: 'UpSampling2D', details: 'size 2x2', row: 7 },
  {
    name: 'conv2d_5',
    className: 'Conv2D',
    details: '256 5x5 filters, 1x1 strides, padding same, ReLU activation',
    row: 8
  },
  { name: 'up_sampling2d_2', className: 'UpSampling2D', details: 'size 2x2', row: 9 },
  {
    name: 'conv2d_6',
    className: 'Conv2D',
    details: '128 5x5 filters, 1x1 strides, padding same, ReLU activation',
    row: 10
  },
  {
    name: 'conv2d_7',
    className: 'Conv2D',
    details: '1 2x2 filters, 1x1 strides, padding same, tanh activation',
    row: 11
  }
]

export const ARCHITECTURE_CONNECTIONS = [
  { from: 'input_2', to: 'multiply_1', corner: 'bottom-left' },
  { from: 'input_3', to: 'embedding_1', corner: 'bottom-right' },
  { from: 'embedding_1', to: 'conv2d_7' }
]
