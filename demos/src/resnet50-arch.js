export const ARCHITECTURE_DIAGRAM = [

  // /////////////////////////////////////////////////////////////////////
  // initial

  {
    name: 'zeropadding2d_1',
    className: 'ZeroPadding2D',
    details: '3x3 padding',
    row: 0,
    col: 0
  },
  {
    name: 'conv1',
    className: 'Convolution2D',
    details: '64 7x7 filters, 2x2 strides, border mode valid',
    row: 1,
    col: 0
  },
  {
    name: 'bn_conv1',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 2,
    col: 0
  },
  {
    name: 'activation_1',
    className: 'Activation',
    details: 'ReLU',
    row: 3,
    col: 0
  },
  {
    name: 'maxpooling2d_1',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, border mode valid',
    row: 4,
    col: 0
  },

  // /////////////////////////////////////////////////////////////////////
  // conv block 2a

  {
    name: 'res2a_branch2a',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode valid',
    row: 5,
    col: 0
  },
  {
    name: 'bn2a_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 6,
    col: 0
  },
  {
    name: 'activation_2',
    className: 'Activation',
    details: 'ReLU',
    row: 7,
    col: 0
  },
  {
    name: 'res2a_branch2b',
    className: 'Convolution2D',
    details: '64 3x3 filters, 1x1 strides, border mode same',
    row: 8,
    col: 0
  },
  {
    name: 'bn2a_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 9,
    col: 0
  },
  {
    name: 'activation_3',
    className: 'Activation',
    details: 'ReLU',
    row: 10,
    col: 0
  },
  {
    name: 'res2a_branch2c',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 11,
    col: 0
  },
  {
    name: 'res2a_branch1',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 5,
    col: 1
  },
  {
    name: 'bn2a_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 12,
    col: 0
  },
  {
    name: 'bn2a_branch1',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 6,
    col: 1
  },
  {
    name: 'merge_1',
    className: 'Merge',
    details: 'sum',
    row: 13,
    col: 1
  },
  {
    name: 'activation_4',
    className: 'Activation',
    details: 'ReLU',
    row: 14,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 2b

  {
    name: 'res2b_branch2a',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode valid',
    row: 15,
    col: 0
  },
  {
    name: 'bn2b_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 16,
    col: 0
  },
  {
    name: 'activation_5',
    className: 'Activation',
    details: 'ReLU',
    row: 17,
    col: 0
  },
  {
    name: 'res2b_branch2b',
    className: 'Convolution2D',
    details: '64 3x3 filters, 1x1 strides, border mode same',
    row: 18,
    col: 0
  },
  {
    name: 'bn2b_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 19,
    col: 0
  },
  {
    name: 'activation_6',
    className: 'Activation',
    details: 'ReLU',
    row: 20,
    col: 0
  },
  {
    name: 'res2b_branch2c',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 21,
    col: 0
  },
  {
    name: 'bn2b_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 22,
    col: 0
  },
  {
    name: 'merge_2',
    className: 'Merge',
    details: 'sum',
    row: 23,
    col: 1
  },
  {
    name: 'activation_7',
    className: 'Activation',
    details: 'ReLU',
    row: 24,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 2c

  {
    name: 'res2c_branch2a',
    className: 'Convolution2D',
    details: '64 1x1 filters, 1x1 strides, border mode valid',
    row: 25,
    col: 0
  },
  {
    name: 'bn2c_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 26,
    col: 0
  },
  {
    name: 'activation_8',
    className: 'Activation',
    details: 'ReLU',
    row: 27,
    col: 0
  },
  {
    name: 'res2c_branch2b',
    className: 'Convolution2D',
    details: '64 3x3 filters, 1x1 strides, border mode same',
    row: 28,
    col: 0
  },
  {
    name: 'bn2c_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 29,
    col: 0
  },
  {
    name: 'activation_9',
    className: 'Activation',
    details: 'ReLU',
    row: 30,
    col: 0
  },
  {
    name: 'res2c_branch2c',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 31,
    col: 0
  },
  {
    name: 'bn2c_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 32,
    col: 0
  },
  {
    name: 'merge_3',
    className: 'Merge',
    details: 'sum',
    row: 33,
    col: 1
  },
  {
    name: 'activation_10',
    className: 'Activation',
    details: 'ReLU',
    row: 34,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // conv block 3a

  {
    name: 'res3a_branch2a',
    className: 'Convolution2D',
    details: '128 1x1 filters, 2x2 strides, border mode valid',
    row: 35,
    col: 0
  },
  {
    name: 'bn3a_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 36,
    col: 0
  },
  {
    name: 'activation_11',
    className: 'Activation',
    details: 'ReLU',
    row: 37,
    col: 0
  },
  {
    name: 'res3a_branch2b',
    className: 'Convolution2D',
    details: '128 3x3 filters, 1x1 strides, border mode same',
    row: 38,
    col: 0
  },
  {
    name: 'bn3a_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 39,
    col: 0
  },
  {
    name: 'activation_12',
    className: 'Activation',
    details: 'ReLU',
    row: 40,
    col: 0
  },
  {
    name: 'res3a_branch2c',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 41,
    col: 0
  },
  {
    name: 'res3a_branch1',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 35,
    col: 1
  },
  {
    name: 'bn3a_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 42,
    col: 0
  },
  {
    name: 'bn3a_branch1',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 36,
    col: 1
  },
  {
    name: 'merge_4',
    className: 'Merge',
    details: 'sum',
    row: 43,
    col: 1
  },
  {
    name: 'activation_13',
    className: 'Activation',
    details: 'ReLU',
    row: 44,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 3b

  {
    name: 'res3b_branch2a',
    className: 'Convolution2D',
    details: '128 1x1 filters, 1x1 strides, border mode valid',
    row: 45,
    col: 0
  },
  {
    name: 'bn3b_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 46,
    col: 0
  },
  {
    name: 'activation_14',
    className: 'Activation',
    details: 'ReLU',
    row: 47,
    col: 0
  },
  {
    name: 'res3b_branch2b',
    className: 'Convolution2D',
    details: '128 3x3 filters, 1x1 strides, border mode same',
    row: 48,
    col: 0
  },
  {
    name: 'bn3b_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 49,
    col: 0
  },
  {
    name: 'activation_15',
    className: 'Activation',
    details: 'ReLU',
    row: 50,
    col: 0
  },
  {
    name: 'res3b_branch2c',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 51,
    col: 0
  },
  {
    name: 'bn3b_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 52,
    col: 0
  },
  {
    name: 'merge_5',
    className: 'Merge',
    details: 'sum',
    row: 53,
    col: 1
  },
  {
    name: 'activation_16',
    className: 'Activation',
    details: 'ReLU',
    row: 54,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 3c

  {
    name: 'res3c_branch2a',
    className: 'Convolution2D',
    details: '128 1x1 filters, 1x1 strides, border mode valid',
    row: 55,
    col: 0
  },
  {
    name: 'bn3c_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 56,
    col: 0
  },
  {
    name: 'activation_17',
    className: 'Activation',
    details: 'ReLU',
    row: 57,
    col: 0
  },
  {
    name: 'res3c_branch2b',
    className: 'Convolution2D',
    details: '128 3x3 filters, 1x1 strides, border mode same',
    row: 58,
    col: 0
  },
  {
    name: 'bn3c_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 59,
    col: 0
  },
  {
    name: 'activation_18',
    className: 'Activation',
    details: 'ReLU',
    row: 60,
    col: 0
  },
  {
    name: 'res3c_branch2c',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 61,
    col: 0
  },
  {
    name: 'bn3c_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 62,
    col: 0
  },
  {
    name: 'merge_6',
    className: 'Merge',
    details: 'sum',
    row: 63,
    col: 1
  },
  {
    name: 'activation_19',
    className: 'Activation',
    details: 'ReLU',
    row: 64,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 3d

  {
    name: 'res3d_branch2a',
    className: 'Convolution2D',
    details: '128 1x1 filters, 1x1 strides, border mode valid',
    row: 65,
    col: 0
  },
  {
    name: 'bn3d_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 66,
    col: 0
  },
  {
    name: 'activation_20',
    className: 'Activation',
    details: 'ReLU',
    row: 67,
    col: 0
  },
  {
    name: 'res3d_branch2b',
    className: 'Convolution2D',
    details: '128 3x3 filters, 1x1 strides, border mode same',
    row: 68,
    col: 0
  },
  {
    name: 'bn3d_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 69,
    col: 0
  },
  {
    name: 'activation_21',
    className: 'Activation',
    details: 'ReLU',
    row: 70,
    col: 0
  },
  {
    name: 'res3d_branch2c',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 71,
    col: 0
  },
  {
    name: 'bn3d_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 72,
    col: 0
  },
  {
    name: 'merge_7',
    className: 'Merge',
    details: 'sum',
    row: 73,
    col: 1
  },
  {
    name: 'activation_22',
    className: 'Activation',
    details: 'ReLU',
    row: 74,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // conv block 4a

  {
    name: 'res4a_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 2x2 strides, border mode valid',
    row: 75,
    col: 0
  },
  {
    name: 'bn4a_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 76,
    col: 0
  },
  {
    name: 'activation_23',
    className: 'Activation',
    details: 'ReLU',
    row: 77,
    col: 0
  },
  {
    name: 'res4a_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 78,
    col: 0
  },
  {
    name: 'bn4a_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 79,
    col: 0
  },
  {
    name: 'activation_24',
    className: 'Activation',
    details: 'ReLU',
    row: 80,
    col: 0
  },
  {
    name: 'res4a_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 81,
    col: 0
  },
  {
    name: 'res4a_branch1',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 2x2 strides, border mode valid',
    row: 75,
    col: 1
  },
  {
    name: 'bn4a_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 82,
    col: 0
  },
  {
    name: 'bn4a_branch1',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 76,
    col: 1
  },
  {
    name: 'merge_8',
    className: 'Merge',
    details: 'sum',
    row: 83,
    col: 1
  },
  {
    name: 'activation_25',
    className: 'Activation',
    details: 'ReLU',
    row: 84,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 4b

  {
    name: 'res4b_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 85,
    col: 0
  },
  {
    name: 'bn4b_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 86,
    col: 0
  },
  {
    name: 'activation_26',
    className: 'Activation',
    details: 'ReLU',
    row: 87,
    col: 0
  },
  {
    name: 'res4b_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 88,
    col: 0
  },
  {
    name: 'bn4b_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 89,
    col: 0
  },
  {
    name: 'activation_27',
    className: 'Activation',
    details: 'ReLU',
    row: 90,
    col: 0
  },
  {
    name: 'res4b_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 91,
    col: 0
  },
  {
    name: 'bn4b_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 92,
    col: 0
  },
  {
    name: 'merge_9',
    className: 'Merge',
    details: 'sum',
    row: 93,
    col: 1
  },
  {
    name: 'activation_28',
    className: 'Activation',
    details: 'ReLU',
    row: 94,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 4c

  {
    name: 'res4c_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 95,
    col: 0
  },
  {
    name: 'bn4c_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 96,
    col: 0
  },
  {
    name: 'activation_29',
    className: 'Activation',
    details: 'ReLU',
    row: 97,
    col: 0
  },
  {
    name: 'res4c_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 98,
    col: 0
  },
  {
    name: 'bn4c_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 99,
    col: 0
  },
  {
    name: 'activation_30',
    className: 'Activation',
    details: 'ReLU',
    row: 100,
    col: 0
  },
  {
    name: 'res4c_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 101,
    col: 0
  },
  {
    name: 'bn4c_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 102,
    col: 0
  },
  {
    name: 'merge_10',
    className: 'Merge',
    details: 'sum',
    row: 103,
    col: 1
  },
  {
    name: 'activation_31',
    className: 'Activation',
    details: 'ReLU',
    row: 104,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 4d

  {
    name: 'res4d_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 105,
    col: 0
  },
  {
    name: 'bn4d_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 106,
    col: 0
  },
  {
    name: 'activation_32',
    className: 'Activation',
    details: 'ReLU',
    row: 107,
    col: 0
  },
  {
    name: 'res4d_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 108,
    col: 0
  },
  {
    name: 'bn4d_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 109,
    col: 0
  },
  {
    name: 'activation_33',
    className: 'Activation',
    details: 'ReLU',
    row: 110,
    col: 0
  },
  {
    name: 'res4d_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 111,
    col: 0
  },
  {
    name: 'bn4d_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 112,
    col: 0
  },
  {
    name: 'merge_11',
    className: 'Merge',
    details: 'sum',
    row: 113,
    col: 1
  },
  {
    name: 'activation_34',
    className: 'Activation',
    details: 'ReLU',
    row: 114,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 4e

  {
    name: 'res4e_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 115,
    col: 0
  },
  {
    name: 'bn4e_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 116,
    col: 0
  },
  {
    name: 'activation_35',
    className: 'Activation',
    details: 'ReLU',
    row: 117,
    col: 0
  },
  {
    name: 'res4e_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 118,
    col: 0
  },
  {
    name: 'bn4e_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 119,
    col: 0
  },
  {
    name: 'activation_36',
    className: 'Activation',
    details: 'ReLU',
    row: 120,
    col: 0
  },
  {
    name: 'res4e_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 121,
    col: 0
  },
  {
    name: 'bn4e_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 122,
    col: 0
  },
  {
    name: 'merge_12',
    className: 'Merge',
    details: 'sum',
    row: 123,
    col: 1
  },
  {
    name: 'activation_37',
    className: 'Activation',
    details: 'ReLU',
    row: 124,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 4f

  {
    name: 'res4f_branch2a',
    className: 'Convolution2D',
    details: '256 1x1 filters, 1x1 strides, border mode valid',
    row: 125,
    col: 0
  },
  {
    name: 'bn4f_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 126,
    col: 0
  },
  {
    name: 'activation_38',
    className: 'Activation',
    details: 'ReLU',
    row: 127,
    col: 0
  },
  {
    name: 'res4f_branch2b',
    className: 'Convolution2D',
    details: '256 3x3 filters, 1x1 strides, border mode same',
    row: 128,
    col: 0
  },
  {
    name: 'bn4f_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 129,
    col: 0
  },
  {
    name: 'activation_39',
    className: 'Activation',
    details: 'ReLU',
    row: 130,
    col: 0
  },
  {
    name: 'res4f_branch2c',
    className: 'Convolution2D',
    details: '1024 1x1 filters, 1x1 strides, border mode valid',
    row: 131,
    col: 0
  },
  {
    name: 'bn4f_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 132,
    col: 0
  },
  {
    name: 'merge_13',
    className: 'Merge',
    details: 'sum',
    row: 133,
    col: 1
  },
  {
    name: 'activation_40',
    className: 'Activation',
    details: 'ReLU',
    row: 134,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // conv block 5a

  {
    name: 'res5a_branch2a',
    className: 'Convolution2D',
    details: '512 1x1 filters, 2x2 strides, border mode valid',
    row: 135,
    col: 0
  },
  {
    name: 'bn5a_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 136,
    col: 0
  },
  {
    name: 'activation_41',
    className: 'Activation',
    details: 'ReLU',
    row: 137,
    col: 0
  },
  {
    name: 'res5a_branch2b',
    className: 'Convolution2D',
    details: '512 3x3 filters, 1x1 strides, border mode same',
    row: 138,
    col: 0
  },
  {
    name: 'bn5a_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 139,
    col: 0
  },
  {
    name: 'activation_42',
    className: 'Activation',
    details: 'ReLU',
    row: 140,
    col: 0
  },
  {
    name: 'res5a_branch2c',
    className: 'Convolution2D',
    details: '2048 1x1 filters, 1x1 strides, border mode valid',
    row: 141,
    col: 0
  },
  {
    name: 'res5a_branch1',
    className: 'Convolution2D',
    details: '2048 1x1 filters, 2x2 strides, border mode valid',
    row: 135,
    col: 1
  },
  {
    name: 'bn5a_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 142,
    col: 0
  },
  {
    name: 'bn5a_branch1',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 136,
    col: 1
  },
  {
    name: 'merge_14',
    className: 'Merge',
    details: 'sum',
    row: 143,
    col: 1
  },
  {
    name: 'activation_43',
    className: 'Activation',
    details: 'ReLU',
    row: 144,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 5b

  {
    name: 'res5b_branch2a',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 145,
    col: 0
  },
  {
    name: 'bn5b_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 146,
    col: 0
  },
  {
    name: 'activation_44',
    className: 'Activation',
    details: 'ReLU',
    row: 147,
    col: 0
  },
  {
    name: 'res5b_branch2b',
    className: 'Convolution2D',
    details: '512 3x3 filters, 1x1 strides, border mode same',
    row: 148,
    col: 0
  },
  {
    name: 'bn5b_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 149,
    col: 0
  },
  {
    name: 'activation_45',
    className: 'Activation',
    details: 'ReLU',
    row: 150,
    col: 0
  },
  {
    name: 'res5b_branch2c',
    className: 'Convolution2D',
    details: '2048 1x1 filters, 1x1 strides, border mode valid',
    row: 151,
    col: 0
  },
  {
    name: 'bn5b_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 152,
    col: 0
  },
  {
    name: 'merge_15',
    className: 'Merge',
    details: 'sum',
    row: 153,
    col: 1
  },
  {
    name: 'activation_46',
    className: 'Activation',
    details: 'ReLU',
    row: 154,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // identity block 5c

  {
    name: 'res5c_branch2a',
    className: 'Convolution2D',
    details: '512 1x1 filters, 1x1 strides, border mode valid',
    row: 155,
    col: 0
  },
  {
    name: 'bn5c_branch2a',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 156,
    col: 0
  },
  {
    name: 'activation_47',
    className: 'Activation',
    details: 'ReLU',
    row: 157,
    col: 0
  },
  {
    name: 'res5c_branch2b',
    className: 'Convolution2D',
    details: '512 3x3 filters, 1x1 strides, border mode same',
    row: 158,
    col: 0
  },
  {
    name: 'bn5c_branch2b',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 159,
    col: 0
  },
  {
    name: 'activation_48',
    className: 'Activation',
    details: 'ReLU',
    row: 160,
    col: 0
  },
  {
    name: 'res5c_branch2c',
    className: 'Convolution2D',
    details: '2048 1x1 filters, 1x1 strides, border mode valid',
    row: 161,
    col: 0
  },
  {
    name: 'bn5c_branch2c',
    className: 'BatchNormalization',
    details: 'feature-wise normalization on channel axis',
    row: 162,
    col: 0
  },
  {
    name: 'merge_16',
    className: 'Merge',
    details: 'sum',
    row: 163,
    col: 1
  },
  {
    name: 'activation_49',
    className: 'Activation',
    details: 'ReLU',
    row: 164,
    col: 1
  },

  // /////////////////////////////////////////////////////////////////////
  // final

  {
    name: 'avg_pool',
    className: 'AveragePooling2D',
    details: '7x7 pool size',
    row: 165,
    col: 1
  },
  {
    name: 'flatten_1',
    className: 'Flatten',
    details: '',
    row: 166,
    col: 1
  },
  {
    name: 'fc1000',
    className: 'Dense',
    details: 'output dimensions 1000, softmax activation',
    row: 167,
    col: 1
  }
]
