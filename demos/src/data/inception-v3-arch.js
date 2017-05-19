export const ARCHITECTURE_DIAGRAM = [
  // /////////////////////////////////////////////////////////////////////
  // initial
  {
    name: 'conv2d_1',
    className: 'Conv2D',
    details: '32 3x3 filters, 2x2 strides, padding valid, ReLU',
    row: 0,
    col: 0
  },
  { name: 'batch_normalization_1', className: 'BatchNormalization', details: 'channel axis features', row: 1, col: 0 },
  {
    name: 'conv2d_2',
    className: 'Conv2D',
    details: '32 3x3 filters, 1x1 strides, padding valid, ReLU',
    row: 2,
    col: 0
  },
  { name: 'batch_normalization_2', className: 'BatchNormalization', details: 'channel axis features', row: 3, col: 0 },
  {
    name: 'conv2d_3',
    className: 'Conv2D',
    details: '64 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 4,
    col: 0
  },
  { name: 'batch_normalization_3', className: 'BatchNormalization', details: 'channel axis features', row: 5, col: 0 },
  {
    name: 'max_pooling2d_1',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, padding valid',
    row: 6,
    col: 0
  },
  {
    name: 'conv2d_4',
    className: 'Conv2D',
    details: '80 1x1 filters, 1x1 strides, padding valid, ReLU',
    row: 7,
    col: 0
  },
  { name: 'batch_normalization_4', className: 'BatchNormalization', details: 'channel axis features', row: 8, col: 0 },
  {
    name: 'conv2d_5',
    className: 'Conv2D',
    details: '192 3x3 filters, 1x1 strides, padding valid, ReLU',
    row: 9,
    col: 0
  },
  { name: 'batch_normalization_5', className: 'BatchNormalization', details: 'channel axis features', row: 10, col: 0 },
  {
    name: 'max_pooling2d_2',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, padding valid',
    row: 11,
    col: 0
  },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 0: 35 x 35 x 256
  {
    name: 'conv2d_9',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 12,
    col: 0
  },
  { name: 'batch_normalization_9', className: 'BatchNormalization', details: 'channel axis features', row: 13, col: 0 },
  {
    name: 'conv2d_7',
    className: 'Conv2D',
    details: '48 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 12,
    col: 1
  },
  {
    name: 'conv2d_10',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 14,
    col: 2
  },
  { name: 'batch_normalization_7', className: 'BatchNormalization', details: 'channel axis features', row: 13, col: 1 },
  {
    name: 'batch_normalization_10',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 15,
    col: 2
  },
  {
    name: 'average_pooling2d_1',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 12,
    col: 3
  },
  {
    name: 'conv2d_6',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 12,
    col: 2
  },
  {
    name: 'conv2d_8',
    className: 'Conv2D',
    details: '64 5x5 filters, 1x1 strides, padding same, ReLU',
    row: 14,
    col: 1
  },
  {
    name: 'conv2d_11',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 16,
    col: 2
  },
  {
    name: 'conv2d_12',
    className: 'Conv2D',
    details: '32 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 13,
    col: 3
  },
  { name: 'batch_normalization_6', className: 'BatchNormalization', details: 'channel axis features', row: 13, col: 2 },
  { name: 'batch_normalization_8', className: 'BatchNormalization', details: 'channel axis features', row: 15, col: 1 },
  {
    name: 'batch_normalization_11',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 17,
    col: 2
  },
  {
    name: 'batch_normalization_12',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 14,
    col: 3
  },
  { name: 'mixed0', className: 'Concatenate', details: 'by channel axis', row: 18, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 1: 35 x 35 x 256
  {
    name: 'conv2d_16',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 19,
    col: 0
  },
  {
    name: 'batch_normalization_16',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 0
  },
  {
    name: 'conv2d_14',
    className: 'Conv2D',
    details: '48 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 19,
    col: 1
  },
  {
    name: 'conv2d_17',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 21,
    col: 2
  },
  {
    name: 'batch_normalization_14',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 1
  },
  {
    name: 'batch_normalization_17',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 22,
    col: 2
  },
  {
    name: 'average_pooling2d_2',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 19,
    col: 3
  },
  {
    name: 'conv2d_13',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 19,
    col: 2
  },
  {
    name: 'conv2d_15',
    className: 'Conv2D',
    details: '64 5x5 filters, 1x1 strides, padding same, ReLU',
    row: 21,
    col: 1
  },
  {
    name: 'conv2d_18',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 23,
    col: 2
  },
  {
    name: 'conv2d_19',
    className: 'Conv2D',
    details: '32 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 20,
    col: 3
  },
  {
    name: 'batch_normalization_13',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 2
  },
  {
    name: 'batch_normalization_15',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 22,
    col: 1
  },
  {
    name: 'batch_normalization_18',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 24,
    col: 2
  },
  {
    name: 'batch_normalization_19',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 21,
    col: 3
  },
  { name: 'mixed1', className: 'Concatenate', details: 'by channel axis', row: 25, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 2: 35 x 35 x 256
  {
    name: 'conv2d_23',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 26,
    col: 0
  },
  {
    name: 'batch_normalization_23',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 0
  },
  {
    name: 'conv2d_21',
    className: 'Conv2D',
    details: '48 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 26,
    col: 1
  },
  {
    name: 'conv2d_24',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 28,
    col: 2
  },
  {
    name: 'batch_normalization_21',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 1
  },
  {
    name: 'batch_normalization_24',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 29,
    col: 2
  },
  {
    name: 'average_pooling2d_3',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 26,
    col: 3
  },
  {
    name: 'conv2d_20',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 26,
    col: 2
  },
  {
    name: 'conv2d_22',
    className: 'Conv2D',
    details: '64 5x5 filters, 1x1 strides, padding same, ReLU',
    row: 28,
    col: 1
  },
  {
    name: 'conv2d_25',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 30,
    col: 2
  },
  {
    name: 'conv2d_26',
    className: 'Conv2D',
    details: '32 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 27,
    col: 3
  },
  {
    name: 'batch_normalization_20',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 2
  },
  {
    name: 'batch_normalization_22',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 29,
    col: 1
  },
  {
    name: 'batch_normalization_25',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 31,
    col: 2
  },
  {
    name: 'batch_normalization_26',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 28,
    col: 3
  },
  { name: 'mixed2', className: 'Concatenate', details: 'by channel axis', row: 32, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 3: 17 x 17 x 768
  {
    name: 'conv2d_28',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 33,
    col: 1
  },
  {
    name: 'batch_normalization_28',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 34,
    col: 1
  },
  {
    name: 'conv2d_29',
    className: 'Conv2D',
    details: '96 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 35,
    col: 1
  },
  {
    name: 'batch_normalization_29',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 36,
    col: 1
  },
  {
    name: 'conv2d_27',
    className: 'Conv2D',
    details: '384 3x3 filters, 2x2 strides, padding valid, ReLU',
    row: 33,
    col: 0
  },
  {
    name: 'conv2d_30',
    className: 'Conv2D',
    details: '96 3x3 filters, 2x2 strides, padding valid, ReLU',
    row: 37,
    col: 1
  },
  {
    name: 'batch_normalization_27',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 34,
    col: 0
  },
  {
    name: 'batch_normalization_30',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 38,
    col: 1
  },
  {
    name: 'max_pooling2d_3',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, padding valid',
    row: 33,
    col: 2
  },
  { name: 'mixed3', className: 'Concatenate', details: 'by channel axis', row: 39, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 4: 17 x 17 x 768
  {
    name: 'conv2d_35',
    className: 'Conv2D',
    details: '128 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 40,
    col: 2
  },
  {
    name: 'batch_normalization_35',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 2
  },
  {
    name: 'conv2d_36',
    className: 'Conv2D',
    details: '128 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 42,
    col: 2
  },
  {
    name: 'batch_normalization_36',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 43,
    col: 2
  },
  {
    name: 'conv2d_32',
    className: 'Conv2D',
    details: '128 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 40,
    col: 1
  },
  {
    name: 'conv2d_37',
    className: 'Conv2D',
    details: '128 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 44,
    col: 2
  },
  {
    name: 'batch_normalization_32',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 1
  },
  {
    name: 'batch_normalization_37',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 45,
    col: 2
  },
  {
    name: 'conv2d_33',
    className: 'Conv2D',
    details: '128 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 42,
    col: 1
  },
  {
    name: 'conv2d_38',
    className: 'Conv2D',
    details: '128 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 46,
    col: 2
  },
  {
    name: 'batch_normalization_33',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 43,
    col: 1
  },
  {
    name: 'batch_normalization_38',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 47,
    col: 2
  },
  {
    name: 'average_pooling2d_4',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 40,
    col: 3
  },
  {
    name: 'conv2d_31',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 40,
    col: 0
  },
  {
    name: 'conv2d_34',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 44,
    col: 1
  },
  {
    name: 'conv2d_39',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 48,
    col: 2
  },
  {
    name: 'conv2d_40',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 41,
    col: 3
  },
  {
    name: 'batch_normalization_31',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 0
  },
  {
    name: 'batch_normalization_34',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 45,
    col: 1
  },
  {
    name: 'batch_normalization_39',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 49,
    col: 2
  },
  {
    name: 'batch_normalization_40',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 42,
    col: 3
  },
  { name: 'mixed4', className: 'Concatenate', details: 'by channel axis', row: 50, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 5: 17 x 17 x 768
  {
    name: 'conv2d_45',
    className: 'Conv2D',
    details: '160 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 51,
    col: 2
  },
  {
    name: 'batch_normalization_45',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 2
  },
  {
    name: 'conv2d_46',
    className: 'Conv2D',
    details: '160 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 53,
    col: 2
  },
  {
    name: 'batch_normalization_46',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 54,
    col: 2
  },
  {
    name: 'conv2d_42',
    className: 'Conv2D',
    details: '160 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 51,
    col: 1
  },
  {
    name: 'conv2d_47',
    className: 'Conv2D',
    details: '160 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 55,
    col: 2
  },
  {
    name: 'batch_normalization_42',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 1
  },
  {
    name: 'batch_normalization_47',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 56,
    col: 2
  },
  {
    name: 'conv2d_43',
    className: 'Conv2D',
    details: '160 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 53,
    col: 1
  },
  {
    name: 'conv2d_48',
    className: 'Conv2D',
    details: '160 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 57,
    col: 2
  },
  {
    name: 'batch_normalization_43',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 54,
    col: 1
  },
  {
    name: 'batch_normalization_48',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 58,
    col: 2
  },
  {
    name: 'average_pooling2d_5',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 51,
    col: 3
  },
  {
    name: 'conv2d_41',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 51,
    col: 0
  },
  {
    name: 'conv2d_44',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 55,
    col: 1
  },
  {
    name: 'conv2d_49',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 59,
    col: 2
  },
  {
    name: 'conv2d_50',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 52,
    col: 3
  },
  {
    name: 'batch_normalization_41',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 0
  },
  {
    name: 'batch_normalization_44',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 56,
    col: 1
  },
  {
    name: 'batch_normalization_49',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 60,
    col: 2
  },
  {
    name: 'batch_normalization_50',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 53,
    col: 3
  },
  { name: 'mixed5', className: 'Concatenate', details: 'by channel axis', row: 61, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 6: 17 x 17 x 768
  {
    name: 'conv2d_55',
    className: 'Conv2D',
    details: '160 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 62,
    col: 2
  },
  {
    name: 'batch_normalization_55',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 2
  },
  {
    name: 'conv2d_56',
    className: 'Conv2D',
    details: '160 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 64,
    col: 2
  },
  {
    name: 'batch_normalization_56',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 65,
    col: 2
  },
  {
    name: 'conv2d_52',
    className: 'Conv2D',
    details: '160 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 62,
    col: 1
  },
  {
    name: 'conv2d_57',
    className: 'Conv2D',
    details: '160 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 66,
    col: 2
  },
  {
    name: 'batch_normalization_52',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 1
  },
  {
    name: 'batch_normalization_57',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 67,
    col: 2
  },
  {
    name: 'conv2d_53',
    className: 'Conv2D',
    details: '160 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 64,
    col: 1
  },
  {
    name: 'conv2d_58',
    className: 'Conv2D',
    details: '160 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 68,
    col: 2
  },
  {
    name: 'batch_normalization_53',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 65,
    col: 1
  },
  {
    name: 'batch_normalization_58',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 69,
    col: 2
  },
  {
    name: 'average_pooling2d_6',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 62,
    col: 3
  },
  {
    name: 'conv2d_51',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 62,
    col: 0
  },
  {
    name: 'conv2d_54',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 66,
    col: 1
  },
  {
    name: 'conv2d_59',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 70,
    col: 2
  },
  {
    name: 'conv2d_60',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 63,
    col: 3
  },
  {
    name: 'batch_normalization_51',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 0
  },
  {
    name: 'batch_normalization_54',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 67,
    col: 1
  },
  {
    name: 'batch_normalization_59',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 71,
    col: 2
  },
  {
    name: 'batch_normalization_60',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 64,
    col: 3
  },
  { name: 'mixed6', className: 'Concatenate', details: 'by channel axis', row: 72, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 7: 17 x 17 x 768
  {
    name: 'conv2d_65',
    className: 'Conv2D',
    details: '160 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 73,
    col: 2
  },
  {
    name: 'batch_normalization_65',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 2
  },
  {
    name: 'conv2d_66',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 75,
    col: 2
  },
  {
    name: 'batch_normalization_66',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 76,
    col: 2
  },
  {
    name: 'conv2d_62',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 73,
    col: 1
  },
  {
    name: 'conv2d_67',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 77,
    col: 2
  },
  {
    name: 'batch_normalization_62',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 1
  },
  {
    name: 'batch_normalization_67',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 78,
    col: 2
  },
  {
    name: 'conv2d_63',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 75,
    col: 1
  },
  {
    name: 'conv2d_68',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 79,
    col: 2
  },
  {
    name: 'batch_normalization_63',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 76,
    col: 1
  },
  {
    name: 'batch_normalization_68',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 80,
    col: 2
  },
  {
    name: 'average_pooling2d_7',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 73,
    col: 3
  },
  {
    name: 'conv2d_61',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 73,
    col: 0
  },
  {
    name: 'conv2d_64',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 77,
    col: 1
  },
  {
    name: 'conv2d_69',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 81,
    col: 2
  },
  {
    name: 'conv2d_70',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 74,
    col: 3
  },
  {
    name: 'batch_normalization_61',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 0
  },
  {
    name: 'batch_normalization_64',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 78,
    col: 1
  },
  {
    name: 'batch_normalization_69',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 82,
    col: 2
  },
  {
    name: 'batch_normalization_70',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 75,
    col: 3
  },
  { name: 'mixed7', className: 'Concatenate', details: 'by channel axis', row: 83, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 8: 8 x 8 x 1280
  {
    name: 'conv2d_73',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 84,
    col: 1
  },
  {
    name: 'batch_normalization_73',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 85,
    col: 1
  },
  {
    name: 'conv2d_74',
    className: 'Conv2D',
    details: '192 1x7 filters, 1x1 strides, padding same, ReLU',
    row: 86,
    col: 1
  },
  {
    name: 'batch_normalization_74',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 87,
    col: 1
  },
  {
    name: 'conv2d_71',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 84,
    col: 0
  },
  {
    name: 'conv2d_75',
    className: 'Conv2D',
    details: '192 7x1 filters, 1x1 strides, padding same, ReLU',
    row: 88,
    col: 1
  },
  {
    name: 'batch_normalization_71',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 85,
    col: 0
  },
  {
    name: 'batch_normalization_75',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 89,
    col: 1
  },
  {
    name: 'conv2d_72',
    className: 'Conv2D',
    details: '320 3x3 filters, 2x2 strides, padding valid, ReLU',
    row: 86,
    col: 0
  },
  {
    name: 'conv2d_76',
    className: 'Conv2D',
    details: '192 3x3 filters, 2x2 strides, padding valid, ReLU',
    row: 90,
    col: 1
  },
  {
    name: 'batch_normalization_72',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 87,
    col: 0
  },
  {
    name: 'batch_normalization_76',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 91,
    col: 1
  },
  {
    name: 'average_pooling2d_8',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 2x2 strides, padding valid',
    row: 84,
    col: 2
  },
  { name: 'mixed8', className: 'Concatenate', details: 'by channel axis', row: 92, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 9: 8 x 8 x 2048
  {
    name: 'conv2d_81',
    className: 'Conv2D',
    details: '448 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 93,
    col: 2
  },
  {
    name: 'batch_normalization_81',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 2
  },
  {
    name: 'conv2d_78',
    className: 'Conv2D',
    details: '384 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 93,
    col: 1
  },
  {
    name: 'conv2d_82',
    className: 'Conv2D',
    details: '384 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 95,
    col: 2
  },
  {
    name: 'batch_normalization_78',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 1
  },
  {
    name: 'batch_normalization_82',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 2
  },
  {
    name: 'conv2d_79',
    className: 'Conv2D',
    details: '384 1x3 filters, 1x1 strides, padding same, ReLU',
    row: 95,
    col: 1,
    subcol: 0
  },
  {
    name: 'conv2d_80',
    className: 'Conv2D',
    details: '384 3x1 filters, 1x1 strides, padding same, ReLU',
    row: 95,
    col: 1,
    subcol: 1
  },
  {
    name: 'conv2d_83',
    className: 'Conv2D',
    details: '384 1x3 filters, 1x1 strides, padding same, ReLU',
    row: 97,
    col: 2,
    subcol: 0
  },
  {
    name: 'conv2d_84',
    className: 'Conv2D',
    details: '384 3x1 filters, 1x1 strides, padding same, ReLU',
    row: 97,
    col: 2,
    subcol: 1
  },
  {
    name: 'average_pooling2d_9',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 93,
    col: 3
  },
  {
    name: 'conv2d_77',
    className: 'Conv2D',
    details: '320 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 93,
    col: 0
  },
  {
    name: 'batch_normalization_79',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 1,
    subcol: 0
  },
  {
    name: 'batch_normalization_80',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 1,
    subcol: 1
  },
  {
    name: 'batch_normalization_83',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 98,
    col: 2,
    subcol: 0
  },
  {
    name: 'batch_normalization_84',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 98,
    col: 2,
    subcol: 1
  },
  {
    name: 'conv2d_85',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 94,
    col: 3
  },
  {
    name: 'batch_normalization_77',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 0
  },
  { name: 'mixed9_0', className: 'Concatenate', details: 'by channel axis', row: 97, col: 1 },
  { name: 'concatenate_1', className: 'Concatenate', details: 'by channel axis', row: 99, col: 2 },
  {
    name: 'batch_normalization_85',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 95,
    col: 3
  },
  { name: 'mixed9', className: 'Concatenate', details: 'by channel axis', row: 100, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // mixed block 10: 8 x 8 x 2048
  {
    name: 'conv2d_90',
    className: 'Conv2D',
    details: '448 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 101,
    col: 2
  },
  {
    name: 'batch_normalization_90',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 2
  },
  {
    name: 'conv2d_87',
    className: 'Conv2D',
    details: '384 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 101,
    col: 1
  },
  {
    name: 'conv2d_91',
    className: 'Conv2D',
    details: '384 3x3 filters, 1x1 strides, padding same, ReLU',
    row: 103,
    col: 2
  },
  {
    name: 'batch_normalization_87',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 1
  },
  {
    name: 'batch_normalization_91',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 2
  },
  {
    name: 'conv2d_88',
    className: 'Conv2D',
    details: '384 1x3 filters, 1x1 strides, padding same, ReLU',
    row: 103,
    col: 1,
    subcol: 0
  },
  {
    name: 'conv2d_89',
    className: 'Conv2D',
    details: '384 3x1 filters, 1x1 strides, padding same, ReLU',
    row: 103,
    col: 1,
    subcol: 1
  },
  {
    name: 'conv2d_92',
    className: 'Conv2D',
    details: '384 1x3 filters, 1x1 strides, padding same, ReLU',
    row: 105,
    col: 2,
    subcol: 0
  },
  {
    name: 'conv2d_93',
    className: 'Conv2D',
    details: '384 3x1 filters, 1x1 strides, padding same, ReLU',
    row: 105,
    col: 2,
    subcol: 1
  },
  {
    name: 'average_pooling2d_10',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, padding same',
    row: 101,
    col: 3
  },
  {
    name: 'conv2d_86',
    className: 'Conv2D',
    details: '320 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 101,
    col: 0
  },
  {
    name: 'batch_normalization_88',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 1,
    subcol: 0
  },
  {
    name: 'batch_normalization_89',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 1,
    subcol: 1
  },
  {
    name: 'batch_normalization_92',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 106,
    col: 2,
    subcol: 0
  },
  {
    name: 'batch_normalization_93',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 106,
    col: 2,
    subcol: 1
  },
  {
    name: 'conv2d_94',
    className: 'Conv2D',
    details: '192 1x1 filters, 1x1 strides, padding same, ReLU',
    row: 102,
    col: 3
  },
  {
    name: 'batch_normalization_86',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 0
  },
  { name: 'mixed9_1', className: 'Concatenate', details: 'by channel axis', row: 105, col: 1 },
  { name: 'concatenate_2', className: 'Concatenate', details: 'by channel axis', row: 107, col: 2 },
  {
    name: 'batch_normalization_94',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 103,
    col: 3
  },
  { name: 'mixed10', className: 'Concatenate', details: 'by channel axis', row: 108, col: 3 },
  // /////////////////////////////////////////////////////////////////////
  // final
  {
    name: 'avg_pool',
    className: 'AveragePooling2D',
    details: '8x8 pool size, 8x8 strides, padding valid',
    row: 109,
    col: 3
  },
  { name: 'flatten', className: 'Flatten', details: '', row: 110, col: 3 },
  { name: 'predictions', className: 'Dense', details: 'output dimensions 1000, softmax activation', row: 111, col: 3 }
]

export const ARCHITECTURE_CONNECTIONS = [
  // main
  { from: 'average_pooling2d_1', to: 'predictions' },
  { from: 'conv2d_1', to: 'max_pooling2d_2' },
  // mixed block 0: 35 x 35 x 256
  { from: 'max_pooling2d_2', to: 'batch_normalization_9' },
  { from: 'max_pooling2d_2', to: 'batch_normalization_11', corner: 'top-right' },
  { from: 'max_pooling2d_2', to: 'batch_normalization_8', corner: 'top-right' },
  { from: 'max_pooling2d_2', to: 'average_pooling2d_1', corner: 'top-right' },
  { from: 'batch_normalization_9', to: 'mixed0', corner: 'bottom-left' },
  { from: 'batch_normalization_11', to: 'mixed0', corner: 'bottom-left' },
  { from: 'batch_normalization_8', to: 'mixed0', corner: 'bottom-left' },
  // mixed block 1: 35 x 35 x 256
  { from: 'mixed0', to: 'conv2d_16', corner: 'bottom-right' },
  { from: 'mixed0', to: 'conv2d_14', corner: 'bottom-right' },
  { from: 'mixed0', to: 'conv2d_13', corner: 'bottom-right' },
  { from: 'conv2d_16', to: 'mixed1', corner: 'bottom-left' },
  { from: 'conv2d_14', to: 'mixed1', corner: 'bottom-left' },
  { from: 'conv2d_13', to: 'mixed1', corner: 'bottom-left' },
  // mixed block 2: 35 x 35 x 256
  { from: 'mixed1', to: 'conv2d_23', corner: 'bottom-right' },
  { from: 'mixed1', to: 'conv2d_21', corner: 'bottom-right' },
  { from: 'mixed1', to: 'conv2d_20', corner: 'bottom-right' },
  { from: 'conv2d_23', to: 'mixed2', corner: 'bottom-left' },
  { from: 'conv2d_21', to: 'mixed2', corner: 'bottom-left' },
  { from: 'conv2d_20', to: 'mixed2', corner: 'bottom-left' },
  // mixed block 3: 17 x 17 x 768
  { from: 'mixed2', to: 'conv2d_27', corner: 'bottom-right' },
  { from: 'mixed2', to: 'conv2d_28', corner: 'bottom-right' },
  { from: 'mixed2', to: 'max_pooling2d_3', corner: 'bottom-right' },
  { from: 'conv2d_27', to: 'mixed3', corner: 'bottom-left' },
  { from: 'conv2d_28', to: 'mixed3', corner: 'bottom-left' },
  { from: 'max_pooling2d_3', to: 'mixed3', corner: 'bottom-left' },
  // mixed block 4: 17 x 17 x 768
  { from: 'mixed3', to: 'conv2d_31', corner: 'bottom-right' },
  { from: 'mixed3', to: 'conv2d_32', corner: 'bottom-right' },
  { from: 'mixed3', to: 'conv2d_35', corner: 'bottom-right' },
  { from: 'conv2d_31', to: 'mixed4', corner: 'bottom-left' },
  { from: 'conv2d_32', to: 'mixed4', corner: 'bottom-left' },
  { from: 'conv2d_35', to: 'mixed4', corner: 'bottom-left' },
  // mixed block 5: 17 x 17 x 768
  { from: 'mixed4', to: 'conv2d_41', corner: 'bottom-right' },
  { from: 'mixed4', to: 'conv2d_42', corner: 'bottom-right' },
  { from: 'mixed4', to: 'conv2d_45', corner: 'bottom-right' },
  { from: 'conv2d_41', to: 'mixed5', corner: 'bottom-left' },
  { from: 'conv2d_42', to: 'mixed5', corner: 'bottom-left' },
  { from: 'conv2d_45', to: 'mixed5', corner: 'bottom-left' },
  // mixed block 6: 17 x 17 x 768
  { from: 'mixed5', to: 'conv2d_51', corner: 'bottom-right' },
  { from: 'mixed5', to: 'conv2d_52', corner: 'bottom-right' },
  { from: 'mixed5', to: 'conv2d_55', corner: 'bottom-right' },
  { from: 'conv2d_51', to: 'mixed6', corner: 'bottom-left' },
  { from: 'conv2d_52', to: 'mixed6', corner: 'bottom-left' },
  { from: 'conv2d_55', to: 'mixed6', corner: 'bottom-left' },
  // mixed block 7: 17 x 17 x 768
  { from: 'mixed6', to: 'conv2d_61', corner: 'bottom-right' },
  { from: 'mixed6', to: 'conv2d_62', corner: 'bottom-right' },
  { from: 'mixed6', to: 'conv2d_65', corner: 'bottom-right' },
  { from: 'conv2d_61', to: 'mixed7', corner: 'bottom-left' },
  { from: 'conv2d_62', to: 'mixed7', corner: 'bottom-left' },
  { from: 'conv2d_65', to: 'mixed7', corner: 'bottom-left' },
  // mixed block 8: 8 x 8 x 1280
  { from: 'mixed7', to: 'conv2d_71', corner: 'bottom-right' },
  { from: 'mixed7', to: 'conv2d_73', corner: 'bottom-right' },
  { from: 'mixed7', to: 'average_pooling2d_8', corner: 'bottom-right' },
  { from: 'conv2d_71', to: 'mixed8', corner: 'bottom-left' },
  { from: 'conv2d_73', to: 'mixed8', corner: 'bottom-left' },
  { from: 'average_pooling2d_8', to: 'mixed8', corner: 'bottom-left' },
  // mixed block 9: 8 x 8 x 2048
  { from: 'mixed8', to: 'conv2d_77', corner: 'bottom-right' },
  { from: 'mixed8', to: 'conv2d_78', corner: 'bottom-right' },
  { from: 'mixed8', to: 'conv2d_81', corner: 'bottom-right' },
  { from: 'conv2d_77', to: 'mixed9', corner: 'bottom-left' },
  { from: 'conv2d_78', to: 'batch_normalization_78' },
  { from: 'batch_normalization_78', to: 'conv2d_79' },
  { from: 'batch_normalization_78', to: 'conv2d_80' },
  { from: 'conv2d_79', to: 'batch_normalization_79' },
  { from: 'conv2d_80', to: 'batch_normalization_80' },
  { from: 'batch_normalization_79', to: 'mixed9_0' },
  { from: 'batch_normalization_80', to: 'mixed9_0' },
  { from: 'mixed9_0', to: 'mixed9', corner: 'bottom-left' },
  { from: 'conv2d_81', to: 'batch_normalization_81' },
  { from: 'batch_normalization_81', to: 'conv2d_82' },
  { from: 'conv2d_82', to: 'batch_normalization_82' },
  { from: 'batch_normalization_82', to: 'conv2d_83' },
  { from: 'batch_normalization_82', to: 'conv2d_84' },
  { from: 'conv2d_83', to: 'batch_normalization_83' },
  { from: 'conv2d_84', to: 'batch_normalization_84' },
  { from: 'batch_normalization_83', to: 'concatenate_1' },
  { from: 'batch_normalization_84', to: 'concatenate_1' },
  { from: 'concatenate_1', to: 'mixed9', corner: 'bottom-left' },
  // mixed block 10: 8 x 8 x 2048
  { from: 'mixed9', to: 'conv2d_86', corner: 'bottom-right' },
  { from: 'mixed9', to: 'conv2d_87', corner: 'bottom-right' },
  { from: 'mixed9', to: 'conv2d_90', corner: 'bottom-right' },
  { from: 'conv2d_86', to: 'mixed10', corner: 'bottom-left' },
  { from: 'conv2d_87', to: 'batch_normalization_87' },
  { from: 'batch_normalization_87', to: 'conv2d_88' },
  { from: 'batch_normalization_87', to: 'conv2d_89' },
  { from: 'conv2d_88', to: 'batch_normalization_88' },
  { from: 'conv2d_89', to: 'batch_normalization_89' },
  { from: 'batch_normalization_88', to: 'mixed9_1' },
  { from: 'batch_normalization_89', to: 'mixed9_1' },
  { from: 'mixed9_1', to: 'mixed10', corner: 'bottom-left' },
  { from: 'conv2d_90', to: 'batch_normalization_90' },
  { from: 'batch_normalization_90', to: 'conv2d_91' },
  { from: 'conv2d_91', to: 'batch_normalization_91' },
  { from: 'batch_normalization_91', to: 'conv2d_92' },
  { from: 'batch_normalization_91', to: 'conv2d_93' },
  { from: 'conv2d_92', to: 'batch_normalization_92' },
  { from: 'conv2d_93', to: 'batch_normalization_93' },
  { from: 'batch_normalization_92', to: 'concatenate_2' },
  { from: 'batch_normalization_93', to: 'concatenate_2' },
  { from: 'concatenate_2', to: 'mixed10', corner: 'bottom-left' }
]
