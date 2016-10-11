export const ARCHITECTURE_DIAGRAM = [

  // /////////////////////////////////////////////////////////////////////
  // initial

  {
    name: 'convolution2d_1',
    className: 'Convolution2D',
    details: '32 3x3 filters, 2x2 strides, border mode valid, ReLU',
    row: 0,
    col: 0
  },
  {
    name: 'batchnormalization_1',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 1,
    col: 0
  },
  {
    name: 'convolution2d_2',
    className: 'Convolution2D',
    details: '32 3x3 filters, 1x1 strides, border mode valid, ReLU',
    row: 2,
    col: 0
  },
  {
    name: 'batchnormalization_2',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 3,
    col: 0
  },
  {
    name: 'convolution2d_3',
    className: 'Convolution2D',
    details: '64 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 4,
    col: 0
  },
  {
    name: 'batchnormalization_3',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 5,
    col: 0
  },
  {
    name: 'maxpooling2d_1',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, border mode valid',
    row: 6,
    col: 0
  },
  {
    name: 'convolution2d_4',
    className: 'Convolution2D',
    details: '80 1x1 filters, 1x1 strides, border mode valid, ReLU',
    row: 7,
    col: 0
  },
  {
    name: 'batchnormalization_4',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 8,
    col: 0
  },
  {
    name: 'convolution2d_5',
    className: 'Convolution2D',
    details: '192 3x3 filters, 1x1 strides, border mode valid, ReLU',
    row: 9,
    col: 0
  },
  {
    name: 'batchnormalization_5',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 10,
    col: 0
  },
  {
    name: 'maxpooling2d_2',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, border mode valid',
    row: 11,
    col: 0
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 0: 35 x 35 x 256

  {
    name: 'convolution2d_9',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 12,
    col: 0
  },
  {
    name: 'batchnormalization_9',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 13,
    col: 0
  },
  {
    name: 'convolution2d_7',
    className: 'Convolution2D',
    details: '48 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 12,
    col: 1
  },
  {
    name: 'convolution2d_10',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 14,
    col: 2
  },
  {
    name: 'batchnormalization_7',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 13,
    col: 1
  },
  {
    name: 'batchnormalization_10',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 15,
    col: 2
  },
  {
    name: 'averagepooling2d_1',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 12,
    col: 3
  },
  {
    name: 'convolution2d_6',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 12,
    col: 2
  },
  {
    name: 'convolution2d_8',
    className: 'Convolution2D',
    details: '64 5x5 filters, 1x1 strides, border mode same, ReLU',
    row: 14,
    col: 1
  },
  {
    name: 'convolution2d_11',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 16,
    col: 2
  },
  {
    name: 'convolution2d_12',
    className: 'Convolution2D',
    details: '32 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 13,
    col: 3
  },
  {
    name: 'batchnormalization_6',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 13,
    col: 2
  },
  {
    name: 'batchnormalization_8',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 15,
    col: 1
  },
  {
    name: 'batchnormalization_11',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 17,
    col: 2
  },
  {
    name: 'batchnormalization_12',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 14,
    col: 3
  },
  {
    name: 'mixed0',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 18,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 1: 35 x 35 x 256

  {
    name: 'convolution2d_16',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 19,
    col: 0
  },
  {
    name: 'batchnormalization_16',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 0
  },
  {
    name: 'convolution2d_14',
    className: 'Convolution2D',
    details: '48 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 19,
    col: 1
  },
  {
    name: 'convolution2d_17',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 21,
    col: 2
  },
  {
    name: 'batchnormalization_14',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 1
  },
  {
    name: 'batchnormalization_17',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 22,
    col: 2
  },
  {
    name: 'averagepooling2d_2',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 19,
    col: 3
  },
  {
    name: 'convolution2d_13',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 19,
    col: 2
  },
  {
    name: 'convolution2d_15',
    className: 'Convolution2D',
    details: '64 5x5 filters, 1x1 strides, border mode same, ReLU',
    row: 21,
    col: 1
  },
  {
    name: 'convolution2d_18',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 23,
    col: 2
  },
  {
    name: 'convolution2d_19',
    className: 'Convolution2D',
    details: '32 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 20,
    col: 3
  },
  {
    name: 'batchnormalization_13',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 20,
    col: 2
  },
  {
    name: 'batchnormalization_15',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 22,
    col: 1
  },
  {
    name: 'batchnormalization_18',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 24,
    col: 2
  },
  {
    name: 'batchnormalization_19',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 21,
    col: 3
  },
  {
    name: 'mixed1',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 25,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 2: 35 x 35 x 256

  {
    name: 'convolution2d_23',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 26,
    col: 0
  },
  {
    name: 'batchnormalization_23',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 0
  },
  {
    name: 'convolution2d_21',
    className: 'Convolution2D',
    details: '48 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 26,
    col: 1
  },
  {
    name: 'convolution2d_24',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 28,
    col: 2
  },
  {
    name: 'batchnormalization_21',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 1
  },
  {
    name: 'batchnormalization_24',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 29,
    col: 2
  },
  {
    name: 'averagepooling2d_3',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 26,
    col: 3
  },
  {
    name: 'convolution2d_20',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 26,
    col: 2
  },
  {
    name: 'convolution2d_22',
    className: 'Convolution2D',
    details: '64 5x5 filters, 1x1 strides, border mode same, ReLU',
    row: 28,
    col: 1
  },
  {
    name: 'convolution2d_25',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 30,
    col: 2
  },
  {
    name: 'convolution2d_26',
    className: 'Convolution2D',
    details: '32 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 27,
    col: 3
  },
  {
    name: 'batchnormalization_20',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 27,
    col: 2
  },
  {
    name: 'batchnormalization_22',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 29,
    col: 1
  },
  {
    name: 'batchnormalization_25',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 31,
    col: 2
  },
  {
    name: 'batchnormalization_26',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 28,
    col: 3
  },
  {
    name: 'mixed2',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 32,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 3: 17 x 17 x 768

  {
    name: 'convolution2d_28',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 33,
    col: 1
  },
  {
    name: 'batchnormalization_28',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 34,
    col: 1
  },
  {
    name: 'convolution2d_29',
    className: 'Convolution2D',
    details: '96 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 35,
    col: 1
  },
  {
    name: 'batchnormalization_29',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 36,
    col: 1
  },
  {
    name: 'convolution2d_27',
    className: 'Convolution2D',
    details: '384 3x3 filters, 2x2 strides, border mode valid, ReLU',
    row: 33,
    col: 0
  },
  {
    name: 'convolution2d_30',
    className: 'Convolution2D',
    details: '96 3x3 filters, 2x2 strides, border mode valid, ReLU',
    row: 37,
    col: 1
  },
  {
    name: 'batchnormalization_27',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 34,
    col: 0
  },
  {
    name: 'batchnormalization_30',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 38,
    col: 1
  },
  {
    name: 'maxpooling2d_3',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, border mode valid',
    row: 33,
    col: 2
  },
  {
    name: 'mixed3',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 39,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 4: 17 x 17 x 768

  {
    name: 'convolution2d_35',
    className: 'Convolution2D',
    details: '128 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 40,
    col: 2
  },
  {
    name: 'batchnormalization_35',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 2
  },
  {
    name: 'convolution2d_36',
    className: 'Convolution2D',
    details: '128 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 42,
    col: 2
  },
  {
    name: 'batchnormalization_36',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 43,
    col: 2
  },
  {
    name: 'convolution2d_32',
    className: 'Convolution2D',
    details: '128 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 40,
    col: 1
  },
  {
    name: 'convolution2d_37',
    className: 'Convolution2D',
    details: '128 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 44,
    col: 2
  },
  {
    name: 'batchnormalization_32',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 1
  },
  {
    name: 'batchnormalization_37',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 45,
    col: 2
  },
  {
    name: 'convolution2d_33',
    className: 'Convolution2D',
    details: '128 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 42,
    col: 1
  },
  {
    name: 'convolution2d_38',
    className: 'Convolution2D',
    details: '128 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 46,
    col: 2
  },
  {
    name: 'batchnormalization_33',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 43,
    col: 1
  },
  {
    name: 'batchnormalization_38',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 47,
    col: 2
  },
  {
    name: 'averagepooling2d_4',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 40,
    col: 3
  },
  {
    name: 'convolution2d_31',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 40,
    col: 0
  },
  {
    name: 'convolution2d_34',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 44,
    col: 1
  },
  {
    name: 'convolution2d_39',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 48,
    col: 2
  },
  {
    name: 'convolution2d_40',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 41,
    col: 3
  },
  {
    name: 'batchnormalization_31',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 41,
    col: 0
  },
  {
    name: 'batchnormalization_34',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 45,
    col: 1
  },
  {
    name: 'batchnormalization_39',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 49,
    col: 2
  },
  {
    name: 'batchnormalization_40',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 42,
    col: 3
  },
  {
    name: 'mixed4',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 50,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 5: 17 x 17 x 768

  {
    name: 'convolution2d_45',
    className: 'Convolution2D',
    details: '160 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 51,
    col: 2
  },
  {
    name: 'batchnormalization_45',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 2
  },
  {
    name: 'convolution2d_46',
    className: 'Convolution2D',
    details: '160 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 53,
    col: 2
  },
  {
    name: 'batchnormalization_46',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 54,
    col: 2
  },
  {
    name: 'convolution2d_42',
    className: 'Convolution2D',
    details: '160 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 51,
    col: 1
  },
  {
    name: 'convolution2d_47',
    className: 'Convolution2D',
    details: '160 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 55,
    col: 2
  },
  {
    name: 'batchnormalization_42',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 1
  },
  {
    name: 'batchnormalization_47',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 56,
    col: 2
  },
  {
    name: 'convolution2d_43',
    className: 'Convolution2D',
    details: '160 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 53,
    col: 1
  },
  {
    name: 'convolution2d_48',
    className: 'Convolution2D',
    details: '160 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 57,
    col: 2
  },
  {
    name: 'batchnormalization_43',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 54,
    col: 1
  },
  {
    name: 'batchnormalization_48',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 58,
    col: 2
  },
  {
    name: 'averagepooling2d_5',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 51,
    col: 3
  },
  {
    name: 'convolution2d_41',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 51,
    col: 0
  },
  {
    name: 'convolution2d_44',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 55,
    col: 1
  },
  {
    name: 'convolution2d_49',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 59,
    col: 2
  },
  {
    name: 'convolution2d_50',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 52,
    col: 3
  },
  {
    name: 'batchnormalization_41',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 52,
    col: 0
  },
  {
    name: 'batchnormalization_44',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 56,
    col: 1
  },
  {
    name: 'batchnormalization_49',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 60,
    col: 2
  },
  {
    name: 'batchnormalization_50',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 53,
    col: 3
  },
  {
    name: 'mixed5',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 61,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 6: 17 x 17 x 768

  {
    name: 'convolution2d_55',
    className: 'Convolution2D',
    details: '160 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 62,
    col: 2
  },
  {
    name: 'batchnormalization_55',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 2
  },
  {
    name: 'convolution2d_56',
    className: 'Convolution2D',
    details: '160 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 64,
    col: 2
  },
  {
    name: 'batchnormalization_56',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 65,
    col: 2
  },
  {
    name: 'convolution2d_52',
    className: 'Convolution2D',
    details: '160 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 62,
    col: 1
  },
  {
    name: 'convolution2d_57',
    className: 'Convolution2D',
    details: '160 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 66,
    col: 2
  },
  {
    name: 'batchnormalization_52',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 1
  },
  {
    name: 'batchnormalization_57',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 67,
    col: 2
  },
  {
    name: 'convolution2d_53',
    className: 'Convolution2D',
    details: '160 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 64,
    col: 1
  },
  {
    name: 'convolution2d_58',
    className: 'Convolution2D',
    details: '160 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 68,
    col: 2
  },
  {
    name: 'batchnormalization_53',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 65,
    col: 1
  },
  {
    name: 'batchnormalization_58',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 69,
    col: 2
  },
  {
    name: 'averagepooling2d_6',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 62,
    col: 3
  },
  {
    name: 'convolution2d_51',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 62,
    col: 0
  },
  {
    name: 'convolution2d_54',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 66,
    col: 1
  },
  {
    name: 'convolution2d_59',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 70,
    col: 2
  },
  {
    name: 'convolution2d_60',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 63,
    col: 3
  },
  {
    name: 'batchnormalization_51',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 63,
    col: 0
  },
  {
    name: 'batchnormalization_54',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 67,
    col: 1
  },
  {
    name: 'batchnormalization_59',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 71,
    col: 2
  },
  {
    name: 'batchnormalization_60',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 64,
    col: 3
  },
  {
    name: 'mixed6',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 72,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 7: 17 x 17 x 768

  {
    name: 'convolution2d_65',
    className: 'Convolution2D',
    details: '160 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 73,
    col: 2
  },
  {
    name: 'batchnormalization_65',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 2
  },
  {
    name: 'convolution2d_66',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 75,
    col: 2
  },
  {
    name: 'batchnormalization_66',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 76,
    col: 2
  },
  {
    name: 'convolution2d_62',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 73,
    col: 1
  },
  {
    name: 'convolution2d_67',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 77,
    col: 2
  },
  {
    name: 'batchnormalization_62',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 1
  },
  {
    name: 'batchnormalization_67',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 78,
    col: 2
  },
  {
    name: 'convolution2d_63',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 75,
    col: 1
  },
  {
    name: 'convolution2d_68',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 79,
    col: 2
  },
  {
    name: 'batchnormalization_63',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 76,
    col: 1
  },
  {
    name: 'batchnormalization_68',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 80,
    col: 2
  },
  {
    name: 'averagepooling2d_7',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 73,
    col: 3
  },
  {
    name: 'convolution2d_61',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 73,
    col: 0
  },
  {
    name: 'convolution2d_64',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 77,
    col: 1
  },
  {
    name: 'convolution2d_69',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 81,
    col: 2
  },
  {
    name: 'convolution2d_70',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 74,
    col: 3
  },
  {
    name: 'batchnormalization_61',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 74,
    col: 0
  },
  {
    name: 'batchnormalization_64',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 78,
    col: 1
  },
  {
    name: 'batchnormalization_69',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 82,
    col: 2
  },
  {
    name: 'batchnormalization_70',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 75,
    col: 3
  },
  {
    name: 'mixed7',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 83,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 8: 8 x 8 x 1280

  {
    name: 'convolution2d_73',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 84,
    col: 1
  },
  {
    name: 'batchnormalization_73',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 85,
    col: 1
  },
  {
    name: 'convolution2d_74',
    className: 'Convolution2D',
    details: '192 1x7 filters, 1x1 strides, border mode same, ReLU',
    row: 86,
    col: 1
  },
  {
    name: 'batchnormalization_74',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 87,
    col: 1
  },
  {
    name: 'convolution2d_71',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 84,
    col: 0
  },
  {
    name: 'convolution2d_75',
    className: 'Convolution2D',
    details: '192 7x1 filters, 1x1 strides, border mode same, ReLU',
    row: 88,
    col: 1
  },
  {
    name: 'batchnormalization_71',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 85,
    col: 0
  },
  {
    name: 'batchnormalization_75',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 89,
    col: 1
  },
  {
    name: 'convolution2d_72',
    className: 'Convolution2D',
    details: '320 3x3 filters, 2x2 strides, border mode valid, ReLU',
    row: 86,
    col: 0
  },
  {
    name: 'convolution2d_76',
    className: 'Convolution2D',
    details: '192 3x3 filters, 2x2 strides, border mode valid, ReLU',
    row: 90,
    col: 1
  },
  {
    name: 'batchnormalization_72',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 87,
    col: 0
  },
  {
    name: 'batchnormalization_76',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 91,
    col: 1
  },
  {
    name: 'averagepooling2d_8',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 2x2 strides, border mode valid',
    row: 84,
    col: 2
  },
  {
    name: 'mixed8',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 92,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 9: 8 x 8 x 2048

  {
    name: 'convolution2d_81',
    className: 'Convolution2D',
    details: '448 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 93,
    col: 2
  },
  {
    name: 'batchnormalization_81',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 2
  },
  {
    name: 'convolution2d_78',
    className: 'Convolution2D',
    details: '384 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 93,
    col: 1
  },
  {
    name: 'convolution2d_82',
    className: 'Convolution2D',
    details: '384 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 95,
    col: 2
  },
  {
    name: 'batchnormalization_78',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 1
  },
  {
    name: 'batchnormalization_82',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 2
  },
  {
    name: 'convolution2d_79',
    className: 'Convolution2D',
    details: '384 1x3 filters, 1x1 strides, border mode same, ReLU',
    row: 95,
    col: 1,
    subcol: 0
  },
  {
    name: 'convolution2d_80',
    className: 'Convolution2D',
    details: '384 3x1 filters, 1x1 strides, border mode same, ReLU',
    row: 95,
    col: 1,
    subcol: 1
  },
  {
    name: 'convolution2d_83',
    className: 'Convolution2D',
    details: '384 1x3 filters, 1x1 strides, border mode same, ReLU',
    row: 97,
    col: 2,
    subcol: 0
  },
  {
    name: 'convolution2d_84',
    className: 'Convolution2D',
    details: '384 3x1 filters, 1x1 strides, border mode same, ReLU',
    row: 97,
    col: 2,
    subcol: 1
  },
  {
    name: 'averagepooling2d_9',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 93,
    col: 3
  },
  {
    name: 'convolution2d_77',
    className: 'Convolution2D',
    details: '320 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 93,
    col: 0
  },
  {
    name: 'batchnormalization_79',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 1,
    subcol: 0
  },
  {
    name: 'batchnormalization_80',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 96,
    col: 1,
    subcol: 1
  },
  {
    name: 'batchnormalization_83',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 98,
    col: 2,
    subcol: 0
  },
  {
    name: 'batchnormalization_84',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 98,
    col: 2,
    subcol: 1
  },
  {
    name: 'convolution2d_85',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 94,
    col: 3
  },
  {
    name: 'batchnormalization_77',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 94,
    col: 0
  },
  {
    name: 'mixed9_0',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 97,
    col: 1
  },
  {
    name: 'merge_1',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 99,
    col: 2
  },
  {
    name: 'batchnormalization_85',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 95,
    col: 3
  },
  {
    name: 'mixed9',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 100,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // mixed block 10: 8 x 8 x 2048

  {
    name: 'convolution2d_90',
    className: 'Convolution2D',
    details: '448 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 101,
    col: 2
  },
  {
    name: 'batchnormalization_90',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 2
  },
  {
    name: 'convolution2d_87',
    className: 'Convolution2D',
    details: '384 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 101,
    col: 1
  },
  {
    name: 'convolution2d_91',
    className: 'Convolution2D',
    details: '384 3x3 filters, 1x1 strides, border mode same, ReLU',
    row: 103,
    col: 2
  },
  {
    name: 'batchnormalization_87',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 1
  },
  {
    name: 'batchnormalization_91',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 2
  },
  {
    name: 'convolution2d_88',
    className: 'Convolution2D',
    details: '384 1x3 filters, 1x1 strides, border mode same, ReLU',
    row: 103,
    col: 1,
    subcol: 0
  },
  {
    name: 'convolution2d_89',
    className: 'Convolution2D',
    details: '384 3x1 filters, 1x1 strides, border mode same, ReLU',
    row: 103,
    col: 1,
    subcol: 1
  },
  {
    name: 'convolution2d_92',
    className: 'Convolution2D',
    details: '384 1x3 filters, 1x1 strides, border mode same, ReLU',
    row: 105,
    col: 2,
    subcol: 0
  },
  {
    name: 'convolution2d_93',
    className: 'Convolution2D',
    details: '384 3x1 filters, 1x1 strides, border mode same, ReLU',
    row: 105,
    col: 2,
    subcol: 1
  },
  {
    name: 'averagepooling2d_10',
    className: 'AveragePooling2D',
    details: '3x3 pool size, 1x1 strides, border mode same',
    row: 101,
    col: 3
  },
  {
    name: 'convolution2d_86',
    className: 'Convolution2D',
    details: '320 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 101,
    col: 0
  },
  {
    name: 'batchnormalization_88',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 1,
    subcol: 0
  },
  {
    name: 'batchnormalization_89',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 104,
    col: 1,
    subcol: 1
  },
  {
    name: 'batchnormalization_92',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 106,
    col: 2,
    subcol: 0
  },
  {
    name: 'batchnormalization_93',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 106,
    col: 2,
    subcol: 1
  },
  {
    name: 'convolution2d_94',
    className: 'Convolution2D',
    details: '192 1x1 filters, 1x1 strides, border mode same, ReLU',
    row: 102,
    col: 3
  },
  {
    name: 'batchnormalization_86',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 102,
    col: 0
  },
  {
    name: 'mixed9_1',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 105,
    col: 1
  },
  {
    name: 'merge_2',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 107,
    col: 2
  },
  {
    name: 'batchnormalization_94',
    className: 'BatchNormalization',
    details: 'channel axis features',
    row: 103,
    col: 3
  },
  {
    name: 'mixed10',
    className: 'Merge',
    details: 'concat by channel axes',
    row: 108,
    col: 3
  },

  // /////////////////////////////////////////////////////////////////////
  // final

  {
    name: 'avg_pool',
    className: 'AveragePooling2D',
    details: '8x8 pool size, 8x8 strides, border mode valid',
    row: 109,
    col: 3
  },
  {
    name: 'flatten',
    className: 'Flatten',
    details: '',
    row: 110,
    col: 3
  },
  {
    name: 'predictions',
    className: 'Dense',
    details: 'output dimensions 1000, softmax activation',
    row: 111,
    col: 3
  }

]

export const ARCHITECTURE_CONNECTIONS = [

  // main

  {
    from: 'averagepooling2d_1',
    to: 'predictions'
  },
  {
    from: 'convolution2d_1',
    to: 'maxpooling2d_2'
  },

  // mixed block 0: 35 x 35 x 256

  {
    from: 'maxpooling2d_2',
    to: 'batchnormalization_9'
  },
  {
    from: 'maxpooling2d_2',
    to: 'batchnormalization_11',
    corner: 'top-right'
  },
  {
    from: 'maxpooling2d_2',
    to: 'batchnormalization_8',
    corner: 'top-right'
  },
  {
    from: 'maxpooling2d_2',
    to: 'averagepooling2d_1',
    corner: 'top-right'
  },
  {
    from: 'batchnormalization_9',
    to: 'mixed0',
    corner: 'bottom-left'
  },
  {
    from: 'batchnormalization_11',
    to: 'mixed0',
    corner: 'bottom-left'
  },
  {
    from: 'batchnormalization_8',
    to: 'mixed0',
    corner: 'bottom-left'
  },

  // mixed block 1: 35 x 35 x 256

  {
    from: 'mixed0',
    to: 'convolution2d_16',
    corner: 'bottom-right'
  },
  {
    from: 'mixed0',
    to: 'convolution2d_14',
    corner: 'bottom-right'
  },
  {
    from: 'mixed0',
    to: 'convolution2d_13',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_16',
    to: 'mixed1',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_14',
    to: 'mixed1',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_13',
    to: 'mixed1',
    corner: 'bottom-left'
  },

  // mixed block 2: 35 x 35 x 256

  {
    from: 'mixed1',
    to: 'convolution2d_23',
    corner: 'bottom-right'
  },
  {
    from: 'mixed1',
    to: 'convolution2d_21',
    corner: 'bottom-right'
  },
  {
    from: 'mixed1',
    to: 'convolution2d_20',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_23',
    to: 'mixed2',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_21',
    to: 'mixed2',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_20',
    to: 'mixed2',
    corner: 'bottom-left'
  },

  // mixed block 3: 17 x 17 x 768

  {
    from: 'mixed2',
    to: 'convolution2d_27',
    corner: 'bottom-right'
  },
  {
    from: 'mixed2',
    to: 'convolution2d_28',
    corner: 'bottom-right'
  },
  {
    from: 'mixed2',
    to: 'maxpooling2d_3',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_27',
    to: 'mixed3',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_28',
    to: 'mixed3',
    corner: 'bottom-left'
  },
  {
    from: 'maxpooling2d_3',
    to: 'mixed3',
    corner: 'bottom-left'
  },

  // mixed block 4: 17 x 17 x 768

  {
    from: 'mixed3',
    to: 'convolution2d_31',
    corner: 'bottom-right'
  },
  {
    from: 'mixed3',
    to: 'convolution2d_32',
    corner: 'bottom-right'
  },
  {
    from: 'mixed3',
    to: 'convolution2d_35',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_31',
    to: 'mixed4',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_32',
    to: 'mixed4',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_35',
    to: 'mixed4',
    corner: 'bottom-left'
  },

  // mixed block 5: 17 x 17 x 768

  {
    from: 'mixed4',
    to: 'convolution2d_41',
    corner: 'bottom-right'
  },
  {
    from: 'mixed4',
    to: 'convolution2d_42',
    corner: 'bottom-right'
  },
  {
    from: 'mixed4',
    to: 'convolution2d_45',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_41',
    to: 'mixed5',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_42',
    to: 'mixed5',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_45',
    to: 'mixed5',
    corner: 'bottom-left'
  },

  // mixed block 6: 17 x 17 x 768

  {
    from: 'mixed5',
    to: 'convolution2d_51',
    corner: 'bottom-right'
  },
  {
    from: 'mixed5',
    to: 'convolution2d_52',
    corner: 'bottom-right'
  },
  {
    from: 'mixed5',
    to: 'convolution2d_55',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_51',
    to: 'mixed6',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_52',
    to: 'mixed6',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_55',
    to: 'mixed6',
    corner: 'bottom-left'
  },

  // mixed block 7: 17 x 17 x 768

  {
    from: 'mixed6',
    to: 'convolution2d_61',
    corner: 'bottom-right'
  },
  {
    from: 'mixed6',
    to: 'convolution2d_62',
    corner: 'bottom-right'
  },
  {
    from: 'mixed6',
    to: 'convolution2d_65',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_61',
    to: 'mixed7',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_62',
    to: 'mixed7',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_65',
    to: 'mixed7',
    corner: 'bottom-left'
  },

  // mixed block 8: 8 x 8 x 1280

  {
    from: 'mixed7',
    to: 'convolution2d_71',
    corner: 'bottom-right'
  },
  {
    from: 'mixed7',
    to: 'convolution2d_73',
    corner: 'bottom-right'
  },
  {
    from: 'mixed7',
    to: 'averagepooling2d_8',
    corner: 'bottom-right'
  },
  {
    from: 'convolution2d_71',
    to: 'mixed8',
    corner: 'bottom-left'
  },
  {
    from: 'convolution2d_73',
    to: 'mixed8',
    corner: 'bottom-left'
  },
  {
    from: 'averagepooling2d_8',
    to: 'mixed8',
    corner: 'bottom-left'
  },

  // mixed block 9: 8 x 8 x 2048

  {
    from: 'mixed8',
    to: 'convolution2d_77',
    corner: 'bottom-right'
  },
  {
    from: 'mixed8',
    to: 'convolution2d_78',
    corner: 'bottom-right'
  },
  {
    from: 'mixed8',
    to: 'convolution2d_81',
    corner: 'bottom-right'
  },

  {
    from: 'convolution2d_77',
    to: 'mixed9',
    corner: 'bottom-left'
  },

  {
    from: 'convolution2d_78',
    to: 'batchnormalization_78'
  },
  {
    from: 'batchnormalization_78',
    to: 'convolution2d_79'
  },
  {
    from: 'batchnormalization_78',
    to: 'convolution2d_80'
  },
  {
    from: 'convolution2d_79',
    to: 'batchnormalization_79'
  },
  {
    from: 'convolution2d_80',
    to: 'batchnormalization_80'
  },
  {
    from: 'batchnormalization_79',
    to: 'mixed9_0'
  },
  {
    from: 'batchnormalization_80',
    to: 'mixed9_0'
  },
  {
    from: 'mixed9_0',
    to: 'mixed9',
    corner: 'bottom-left'
  },

  {
    from: 'convolution2d_81',
    to: 'batchnormalization_81'
  },
  {
    from: 'batchnormalization_81',
    to: 'convolution2d_82'
  },
  {
    from: 'convolution2d_82',
    to: 'batchnormalization_82'
  },
  {
    from: 'batchnormalization_82',
    to: 'convolution2d_83'
  },
  {
    from: 'batchnormalization_82',
    to: 'convolution2d_84'
  },
  {
    from: 'convolution2d_83',
    to: 'batchnormalization_83'
  },
  {
    from: 'convolution2d_84',
    to: 'batchnormalization_84'
  },
  {
    from: 'batchnormalization_83',
    to: 'merge_1'
  },
  {
    from: 'batchnormalization_84',
    to: 'merge_1'
  },
  {
    from: 'merge_1',
    to: 'mixed9',
    corner: 'bottom-left'
  },

  // mixed block 10: 8 x 8 x 2048

  {
    from: 'mixed9',
    to: 'convolution2d_86',
    corner: 'bottom-right'
  },
  {
    from: 'mixed9',
    to: 'convolution2d_87',
    corner: 'bottom-right'
  },
  {
    from: 'mixed9',
    to: 'convolution2d_90',
    corner: 'bottom-right'
  },

  {
    from: 'convolution2d_86',
    to: 'mixed10',
    corner: 'bottom-left'
  },

  {
    from: 'convolution2d_87',
    to: 'batchnormalization_87'
  },
  {
    from: 'batchnormalization_87',
    to: 'convolution2d_88'
  },
  {
    from: 'batchnormalization_87',
    to: 'convolution2d_89'
  },
  {
    from: 'convolution2d_88',
    to: 'batchnormalization_88'
  },
  {
    from: 'convolution2d_89',
    to: 'batchnormalization_89'
  },
  {
    from: 'batchnormalization_88',
    to: 'mixed9_1'
  },
  {
    from: 'batchnormalization_89',
    to: 'mixed9_1'
  },
  {
    from: 'mixed9_1',
    to: 'mixed10',
    corner: 'bottom-left'
  },

  {
    from: 'convolution2d_90',
    to: 'batchnormalization_90'
  },
  {
    from: 'batchnormalization_90',
    to: 'convolution2d_91'
  },
  {
    from: 'convolution2d_91',
    to: 'batchnormalization_91'
  },
  {
    from: 'batchnormalization_91',
    to: 'convolution2d_92'
  },
  {
    from: 'batchnormalization_91',
    to: 'convolution2d_93'
  },
  {
    from: 'convolution2d_92',
    to: 'batchnormalization_92'
  },
  {
    from: 'convolution2d_93',
    to: 'batchnormalization_93'
  },
  {
    from: 'batchnormalization_92',
    to: 'merge_2'
  },
  {
    from: 'batchnormalization_93',
    to: 'merge_2'
  },
  {
    from: 'merge_2',
    to: 'mixed10',
    corner: 'bottom-left'
  }

]
