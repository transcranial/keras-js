export const ARCHITECTURE_DIAGRAM = [
  {
    className: 'Convolution2D',
    name: 'block1_conv1',
    row: 0,
    details: '32 3x3 filters, 2x2 strides, border mode valid',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block1_conv1_bn',
    row: 1,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block1_conv1_act',
    row: 2,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'Convolution2D',
    name: 'block1_conv2',
    row: 3,
    details: '64 3x3 filters, 1x1 strides, border mode valid',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block1_conv2_bn',
    row: 4,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block1_conv2_act',
    row: 5,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block2_sepconv1',
    row: 6,
    details: '128 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block2_sepconv1_bn',
    row: 7,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block2_sepconv2_act',
    row: 8,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block2_sepconv2',
    row: 9,
    details: '128 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block2_sepconv2_bn',
    row: 10,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Convolution2D',
    name: 'convolution2d_1',
    row: 6,
    details: '128 1x1 filters, 2x2 strides, border mode same',
    col: 1
  },
  {
    className: 'MaxPooling2D',
    name: 'block2_pool',
    row: 11,
    details: '3x3 pool size, 2x2 strides, border mode same',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'batchnormalization_1',
    row: 7,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Merge',
    name: 'merge_1',
    row: 12,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block3_sepconv1_act',
    row: 13,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block3_sepconv1',
    row: 14,
    details: '256 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block3_sepconv1_bn',
    row: 15,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block3_sepconv2_act',
    row: 16,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block3_sepconv2',
    row: 17,
    details: '256 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block3_sepconv2_bn',
    row: 18,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Convolution2D',
    name: 'convolution2d_2',
    row: 13,
    details: '256 1x1 filters, 2x2 strides, border mode same',
    col: 1
  },
  {
    className: 'MaxPooling2D',
    name: 'block3_pool',
    row: 19,
    details: '3x3 pool size, 2x2 strides, border mode same',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'batchnormalization_2',
    row: 14,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Merge',
    name: 'merge_2',
    row: 20,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block4_sepconv1_act',
    row: 21,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block4_sepconv1',
    row: 22,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block4_sepconv1_bn',
    row: 23,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block4_sepconv2_act',
    row: 24,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block4_sepconv2',
    row: 25,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block4_sepconv2_bn',
    row: 26,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Convolution2D',
    name: 'convolution2d_3',
    row: 21,
    details: '728 1x1 filters, 2x2 strides, border mode same',
    col: 1
  },
  {
    className: 'MaxPooling2D',
    name: 'block4_pool',
    row: 27,
    details: '3x3 pool size, 2x2 strides, border mode same',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'batchnormalization_3',
    row: 22,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Merge',
    name: 'merge_3',
    row: 28,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block5_sepconv1_act',
    row: 29,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block5_sepconv1',
    row: 30,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block5_sepconv1_bn',
    row: 31,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block5_sepconv2_act',
    row: 32,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block5_sepconv2',
    row: 33,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block5_sepconv2_bn',
    row: 34,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block5_sepconv3_act',
    row: 35,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block5_sepconv3',
    row: 36,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block5_sepconv3_bn',
    row: 37,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_4',
    row: 38,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block6_sepconv1_act',
    row: 39,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block6_sepconv1',
    row: 40,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block6_sepconv1_bn',
    row: 41,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block6_sepconv2_act',
    row: 42,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block6_sepconv2',
    row: 43,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block6_sepconv2_bn',
    row: 44,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block6_sepconv3_act',
    row: 45,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block6_sepconv3',
    row: 46,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block6_sepconv3_bn',
    row: 47,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_5',
    row: 48,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block7_sepconv1_act',
    row: 49,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block7_sepconv1',
    row: 50,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block7_sepconv1_bn',
    row: 51,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block7_sepconv2_act',
    row: 52,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block7_sepconv2',
    row: 53,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block7_sepconv2_bn',
    row: 54,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block7_sepconv3_act',
    row: 55,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block7_sepconv3',
    row: 56,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block7_sepconv3_bn',
    row: 57,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_6',
    row: 58,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block8_sepconv1_act',
    row: 59,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block8_sepconv1',
    row: 60,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block8_sepconv1_bn',
    row: 61,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block8_sepconv2_act',
    row: 62,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block8_sepconv2',
    row: 63,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block8_sepconv2_bn',
    row: 64,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block8_sepconv3_act',
    row: 65,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block8_sepconv3',
    row: 66,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block8_sepconv3_bn',
    row: 67,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_7',
    row: 68,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block9_sepconv1_act',
    row: 69,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block9_sepconv1',
    row: 70,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block9_sepconv1_bn',
    row: 71,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block9_sepconv2_act',
    row: 72,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block9_sepconv2',
    row: 73,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block9_sepconv2_bn',
    row: 74,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block9_sepconv3_act',
    row: 75,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block9_sepconv3',
    row: 76,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block9_sepconv3_bn',
    row: 77,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_8',
    row: 78,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block10_sepconv1_act',
    row: 79,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block10_sepconv1',
    row: 80,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block10_sepconv1_bn',
    row: 81,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block10_sepconv2_act',
    row: 82,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block10_sepconv2',
    row: 83,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block10_sepconv2_bn',
    row: 84,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block10_sepconv3_act',
    row: 85,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block10_sepconv3',
    row: 86,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block10_sepconv3_bn',
    row: 87,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_9',
    row: 88,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block11_sepconv1_act',
    row: 89,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block11_sepconv1',
    row: 90,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block11_sepconv1_bn',
    row: 91,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block11_sepconv2_act',
    row: 92,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block11_sepconv2',
    row: 93,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block11_sepconv2_bn',
    row: 94,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block11_sepconv3_act',
    row: 95,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block11_sepconv3',
    row: 96,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block11_sepconv3_bn',
    row: 97,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_10',
    row: 98,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block12_sepconv1_act',
    row: 99,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block12_sepconv1',
    row: 100,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block12_sepconv1_bn',
    row: 101,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block12_sepconv2_act',
    row: 102,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block12_sepconv2',
    row: 103,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block12_sepconv2_bn',
    row: 104,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block12_sepconv3_act',
    row: 105,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block12_sepconv3',
    row: 106,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block12_sepconv3_bn',
    row: 107,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Merge',
    name: 'merge_11',
    row: 108,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block13_sepconv1_act',
    row: 109,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block13_sepconv1',
    row: 110,
    details: '728 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block13_sepconv1_bn',
    row: 111,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Activation',
    name: 'block13_sepconv2_act',
    row: 112,
    details: 'ReLU',
    col: 0
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block13_sepconv2',
    row: 113,
    details: '1024 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'block13_sepconv2_bn',
    row: 114,
    details: 'channel axis features',
    col: 0
  },
  {
    className: 'Convolution2D',
    name: 'convolution2d_4',
    row: 109,
    details: '1024 1x1 filters, 2x2 strides, border mode same',
    col: 1
  },
  {
    className: 'MaxPooling2D',
    name: 'block13_pool',
    row: 115,
    details: '3x3 pool size, 2x2 strides, border mode same',
    col: 0
  },
  {
    className: 'BatchNormalization',
    name: 'batchnormalization_4',
    row: 110,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Merge',
    name: 'merge_12',
    row: 116,
    details: 'sum along channel axes',
    col: 1
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block14_sepconv1',
    row: 117,
    details: '1536 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 1
  },
  {
    className: 'BatchNormalization',
    name: 'block14_sepconv1_bn',
    row: 118,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block14_sepconv1_act',
    row: 119,
    details: 'ReLU',
    col: 1
  },
  {
    className: 'SeparableConvolution2D',
    name: 'block14_sepconv2',
    row: 120,
    details: '2048 3x3 filters, 1x1 strides, border mode same, depth multiplier 1',
    col: 1
  },
  {
    className: 'BatchNormalization',
    name: 'block14_sepconv2_bn',
    row: 121,
    details: 'channel axis features',
    col: 1
  },
  {
    className: 'Activation',
    name: 'block14_sepconv2_act',
    row: 122,
    details: 'ReLU',
    col: 1
  },
  {
    className: 'GlobalAveragePooling2D',
    name: 'avg_pool',
    row: 123,
    details: '',
    col: 1
  },
  {
    className: 'Dense',
    name: 'predictions',
    row: 124,
    details: 'output dimensions 1000, softmax activation',
    col: 1
  }
]

export const ARCHITECTURE_CONNECTIONS = [
  {
    from: 'convolution2d_1',
    to: 'predictions'
  },

  // blocks

  {
    from: 'block1_conv1',
    to: 'block2_pool'
  },
  {
    from: 'block3_sepconv1_act',
    to: 'block3_pool'
  },
  {
    from: 'block4_sepconv1_act',
    to: 'block4_pool'
  },
  {
    from: 'block5_sepconv1_act',
    to: 'block5_sepconv3_bn'
  },
  {
    from: 'block6_sepconv1_act',
    to: 'block6_sepconv3_bn'
  },
  {
    from: 'block7_sepconv1_act',
    to: 'block7_sepconv3_bn'
  },
  {
    from: 'block8_sepconv1_act',
    to: 'block8_sepconv3_bn'
  },
  {
    from: 'block9_sepconv1_act',
    to: 'block9_sepconv3_bn'
  },
  {
    from: 'block10_sepconv1_act',
    to: 'block10_sepconv3_bn'
  },
  {
    from: 'block11_sepconv1_act',
    to: 'block11_sepconv3_bn'
  },
  {
    from: 'block12_sepconv1_act',
    to: 'block12_sepconv3_bn'
  },
  {
    from: 'block13_sepconv1_act',
    to: 'block13_pool'
  },

  // connections

  {
    from: 'block1_conv2_act',
    to: 'convolution2d_1',
    corner: 'top-right'
  },
  {
    from: 'block2_pool',
    to: 'merge_1',
    corner: 'top-right'
  },
  {
    from: 'block3_pool',
    to: 'merge_2',
    corner: 'top-right'
  },
  {
    from: 'block4_pool',
    to: 'merge_3',
    corner: 'top-right'
  },
  {
    from: 'block5_sepconv3_bn',
    to: 'merge_4',
    corner: 'top-right'
  },
  {
    from: 'block6_sepconv3_bn',
    to: 'merge_5',
    corner: 'top-right'
  },
  {
    from: 'block7_sepconv3_bn',
    to: 'merge_6',
    corner: 'top-right'
  },
  {
    from: 'block8_sepconv3_bn',
    to: 'merge_7',
    corner: 'top-right'
  },
  {
    from: 'block9_sepconv3_bn',
    to: 'merge_8',
    corner: 'top-right'
  },
  {
    from: 'block10_sepconv3_bn',
    to: 'merge_9',
    corner: 'top-right'
  },
  {
    from: 'block11_sepconv3_bn',
    to: 'merge_10',
    corner: 'top-right'
  },
  {
    from: 'block12_sepconv3_bn',
    to: 'merge_11',
    corner: 'top-right'
  },
  {
    from: 'block13_pool',
    to: 'merge_12',
    corner: 'top-right'
  },
  {
    from: 'merge_1',
    to: 'block3_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_2',
    to: 'block4_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_3',
    to: 'block5_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_4',
    to: 'block6_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_5',
    to: 'block7_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_6',
    to: 'block8_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_7',
    to: 'block9_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_8',
    to: 'block10_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_9',
    to: 'block11_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_10',
    to: 'block12_sepconv1_act',
    corner: 'top-left'
  },
  {
    from: 'merge_11',
    to: 'block13_sepconv1_act',
    corner: 'top-left'
  }
]
