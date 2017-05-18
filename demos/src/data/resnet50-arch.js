export const ARCHITECTURE_DIAGRAM = [
  // /////////////////////////////////////////////////////////////////////
  // initial
  { name: 'zero_padding2d_1', className: 'ZeroPadding2D', details: '3x3 padding', row: 0, col: 0 },
  {
    name: 'conv1',
    className: 'Conv2D',
    details: '64 7x7 filters, 2x2 strides, padding valid',
    row: 1,
    col: 0
  },
  { name: 'bn_conv1', className: 'BatchNormalization', details: 'channel axis features', row: 2, col: 0 },
  { name: 'activation_1', className: 'Activation', details: 'ReLU', row: 3, col: 0 },
  {
    name: 'max_pooling2d_1',
    className: 'MaxPooling2D',
    details: '3x3 pool size, 2x2 strides, padding valid',
    row: 4,
    col: 0
  },
  // /////////////////////////////////////////////////////////////////////
  // conv block 2a
  {
    name: 'res2a_branch2a',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding valid',
    row: 5,
    col: 0
  },
  { name: 'bn2a_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 6, col: 0 },
  { name: 'activation_2', className: 'Activation', details: 'ReLU', row: 7, col: 0 },
  {
    name: 'res2a_branch2b',
    className: 'Conv2D',
    details: '64 3x3 filters, 1x1 strides, padding same',
    row: 8,
    col: 0
  },
  { name: 'bn2a_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 9, col: 0 },
  { name: 'activation_3', className: 'Activation', details: 'ReLU', row: 10, col: 0 },
  {
    name: 'res2a_branch2c',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 11,
    col: 0
  },
  {
    name: 'res2a_branch1',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 5,
    col: 1
  },
  { name: 'bn2a_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 12, col: 0 },
  { name: 'bn2a_branch1', className: 'BatchNormalization', details: 'channel axis features', row: 6, col: 1 },
  { name: 'add_1', className: 'Add', row: 13, col: 1 },
  { name: 'activation_4', className: 'Activation', details: 'ReLU', row: 14, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 2b
  {
    name: 'res2b_branch2a',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding valid',
    row: 15,
    col: 0
  },
  { name: 'bn2b_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 16, col: 0 },
  { name: 'activation_5', className: 'Activation', details: 'ReLU', row: 17, col: 0 },
  {
    name: 'res2b_branch2b',
    className: 'Conv2D',
    details: '64 3x3 filters, 1x1 strides, padding same',
    row: 18,
    col: 0
  },
  { name: 'bn2b_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 19, col: 0 },
  { name: 'activation_6', className: 'Activation', details: 'ReLU', row: 20, col: 0 },
  {
    name: 'res2b_branch2c',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 21,
    col: 0
  },
  { name: 'bn2b_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 22, col: 0 },
  { name: 'add_2', className: 'Add', row: 23, col: 1 },
  { name: 'activation_7', className: 'Activation', details: 'ReLU', row: 24, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 2c
  {
    name: 'res2c_branch2a',
    className: 'Conv2D',
    details: '64 1x1 filters, 1x1 strides, padding valid',
    row: 25,
    col: 0
  },
  { name: 'bn2c_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 26, col: 0 },
  { name: 'activation_8', className: 'Activation', details: 'ReLU', row: 27, col: 0 },
  {
    name: 'res2c_branch2b',
    className: 'Conv2D',
    details: '64 3x3 filters, 1x1 strides, padding same',
    row: 28,
    col: 0
  },
  { name: 'bn2c_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 29, col: 0 },
  { name: 'activation_9', className: 'Activation', details: 'ReLU', row: 30, col: 0 },
  {
    name: 'res2c_branch2c',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 31,
    col: 0
  },
  { name: 'bn2c_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 32, col: 0 },
  { name: 'add_3', className: 'Add', row: 33, col: 1 },
  { name: 'activation_10', className: 'Activation', details: 'ReLU', row: 34, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // conv block 3a
  {
    name: 'res3a_branch2a',
    className: 'Conv2D',
    details: '128 1x1 filters, 2x2 strides, padding valid',
    row: 35,
    col: 0
  },
  { name: 'bn3a_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 36, col: 0 },
  { name: 'activation_11', className: 'Activation', details: 'ReLU', row: 37, col: 0 },
  {
    name: 'res3a_branch2b',
    className: 'Conv2D',
    details: '128 3x3 filters, 1x1 strides, padding same',
    row: 38,
    col: 0
  },
  { name: 'bn3a_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 39, col: 0 },
  { name: 'activation_12', className: 'Activation', details: 'ReLU', row: 40, col: 0 },
  {
    name: 'res3a_branch2c',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 41,
    col: 0
  },
  {
    name: 'res3a_branch1',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 35,
    col: 1
  },
  { name: 'bn3a_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 42, col: 0 },
  { name: 'bn3a_branch1', className: 'BatchNormalization', details: 'channel axis features', row: 36, col: 1 },
  { name: 'add_4', className: 'Add', row: 43, col: 1 },
  { name: 'activation_13', className: 'Activation', details: 'ReLU', row: 44, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 3b
  {
    name: 'res3b_branch2a',
    className: 'Conv2D',
    details: '128 1x1 filters, 1x1 strides, padding valid',
    row: 45,
    col: 0
  },
  { name: 'bn3b_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 46, col: 0 },
  { name: 'activation_14', className: 'Activation', details: 'ReLU', row: 47, col: 0 },
  {
    name: 'res3b_branch2b',
    className: 'Conv2D',
    details: '128 3x3 filters, 1x1 strides, padding same',
    row: 48,
    col: 0
  },
  { name: 'bn3b_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 49, col: 0 },
  { name: 'activation_15', className: 'Activation', details: 'ReLU', row: 50, col: 0 },
  {
    name: 'res3b_branch2c',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 51,
    col: 0
  },
  { name: 'bn3b_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 52, col: 0 },
  { name: 'add_5', className: 'Add', row: 53, col: 1 },
  { name: 'activation_16', className: 'Activation', details: 'ReLU', row: 54, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 3c
  {
    name: 'res3c_branch2a',
    className: 'Conv2D',
    details: '128 1x1 filters, 1x1 strides, padding valid',
    row: 55,
    col: 0
  },
  { name: 'bn3c_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 56, col: 0 },
  { name: 'activation_17', className: 'Activation', details: 'ReLU', row: 57, col: 0 },
  {
    name: 'res3c_branch2b',
    className: 'Conv2D',
    details: '128 3x3 filters, 1x1 strides, padding same',
    row: 58,
    col: 0
  },
  { name: 'bn3c_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 59, col: 0 },
  { name: 'activation_18', className: 'Activation', details: 'ReLU', row: 60, col: 0 },
  {
    name: 'res3c_branch2c',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 61,
    col: 0
  },
  { name: 'bn3c_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 62, col: 0 },
  { name: 'add_6', className: 'Add', row: 63, col: 1 },
  { name: 'activation_19', className: 'Activation', details: 'ReLU', row: 64, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 3d
  {
    name: 'res3d_branch2a',
    className: 'Conv2D',
    details: '128 1x1 filters, 1x1 strides, padding valid',
    row: 65,
    col: 0
  },
  { name: 'bn3d_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 66, col: 0 },
  { name: 'activation_20', className: 'Activation', details: 'ReLU', row: 67, col: 0 },
  {
    name: 'res3d_branch2b',
    className: 'Conv2D',
    details: '128 3x3 filters, 1x1 strides, padding same',
    row: 68,
    col: 0
  },
  { name: 'bn3d_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 69, col: 0 },
  { name: 'activation_21', className: 'Activation', details: 'ReLU', row: 70, col: 0 },
  {
    name: 'res3d_branch2c',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 71,
    col: 0
  },
  { name: 'bn3d_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 72, col: 0 },
  { name: 'add_7', className: 'Add', row: 73, col: 1 },
  { name: 'activation_22', className: 'Activation', details: 'ReLU', row: 74, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // conv block 4a
  {
    name: 'res4a_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 2x2 strides, padding valid',
    row: 75,
    col: 0
  },
  { name: 'bn4a_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 76, col: 0 },
  { name: 'activation_23', className: 'Activation', details: 'ReLU', row: 77, col: 0 },
  {
    name: 'res4a_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 78,
    col: 0
  },
  { name: 'bn4a_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 79, col: 0 },
  { name: 'activation_24', className: 'Activation', details: 'ReLU', row: 80, col: 0 },
  {
    name: 'res4a_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 81,
    col: 0
  },
  {
    name: 'res4a_branch1',
    className: 'Conv2D',
    details: '1024 1x1 filters, 2x2 strides, padding valid',
    row: 75,
    col: 1
  },
  { name: 'bn4a_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 82, col: 0 },
  { name: 'bn4a_branch1', className: 'BatchNormalization', details: 'channel axis features', row: 76, col: 1 },
  { name: 'add_8', className: 'Add', row: 83, col: 1 },
  { name: 'activation_25', className: 'Activation', details: 'ReLU', row: 84, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 4b
  {
    name: 'res4b_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 85,
    col: 0
  },
  { name: 'bn4b_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 86, col: 0 },
  { name: 'activation_26', className: 'Activation', details: 'ReLU', row: 87, col: 0 },
  {
    name: 'res4b_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 88,
    col: 0
  },
  { name: 'bn4b_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 89, col: 0 },
  { name: 'activation_27', className: 'Activation', details: 'ReLU', row: 90, col: 0 },
  {
    name: 'res4b_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 91,
    col: 0
  },
  { name: 'bn4b_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 92, col: 0 },
  { name: 'add_9', className: 'Add', row: 93, col: 1 },
  { name: 'activation_28', className: 'Activation', details: 'ReLU', row: 94, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 4c
  {
    name: 'res4c_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 95,
    col: 0
  },
  { name: 'bn4c_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 96, col: 0 },
  { name: 'activation_29', className: 'Activation', details: 'ReLU', row: 97, col: 0 },
  {
    name: 'res4c_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 98,
    col: 0
  },
  { name: 'bn4c_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 99, col: 0 },
  { name: 'activation_30', className: 'Activation', details: 'ReLU', row: 100, col: 0 },
  {
    name: 'res4c_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 101,
    col: 0
  },
  { name: 'bn4c_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 102, col: 0 },
  { name: 'add_10', className: 'Add', row: 103, col: 1 },
  { name: 'activation_31', className: 'Activation', details: 'ReLU', row: 104, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 4d
  {
    name: 'res4d_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 105,
    col: 0
  },
  { name: 'bn4d_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 106, col: 0 },
  { name: 'activation_32', className: 'Activation', details: 'ReLU', row: 107, col: 0 },
  {
    name: 'res4d_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 108,
    col: 0
  },
  { name: 'bn4d_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 109, col: 0 },
  { name: 'activation_33', className: 'Activation', details: 'ReLU', row: 110, col: 0 },
  {
    name: 'res4d_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 111,
    col: 0
  },
  { name: 'bn4d_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 112, col: 0 },
  { name: 'add_11', className: 'Add', row: 113, col: 1 },
  { name: 'activation_34', className: 'Activation', details: 'ReLU', row: 114, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 4e
  {
    name: 'res4e_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 115,
    col: 0
  },
  { name: 'bn4e_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 116, col: 0 },
  { name: 'activation_35', className: 'Activation', details: 'ReLU', row: 117, col: 0 },
  {
    name: 'res4e_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 118,
    col: 0
  },
  { name: 'bn4e_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 119, col: 0 },
  { name: 'activation_36', className: 'Activation', details: 'ReLU', row: 120, col: 0 },
  {
    name: 'res4e_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 121,
    col: 0
  },
  { name: 'bn4e_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 122, col: 0 },
  { name: 'add_12', className: 'Add', row: 123, col: 1 },
  { name: 'activation_37', className: 'Activation', details: 'ReLU', row: 124, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 4f
  {
    name: 'res4f_branch2a',
    className: 'Conv2D',
    details: '256 1x1 filters, 1x1 strides, padding valid',
    row: 125,
    col: 0
  },
  { name: 'bn4f_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 126, col: 0 },
  { name: 'activation_38', className: 'Activation', details: 'ReLU', row: 127, col: 0 },
  {
    name: 'res4f_branch2b',
    className: 'Conv2D',
    details: '256 3x3 filters, 1x1 strides, padding same',
    row: 128,
    col: 0
  },
  { name: 'bn4f_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 129, col: 0 },
  { name: 'activation_39', className: 'Activation', details: 'ReLU', row: 130, col: 0 },
  {
    name: 'res4f_branch2c',
    className: 'Conv2D',
    details: '1024 1x1 filters, 1x1 strides, padding valid',
    row: 131,
    col: 0
  },
  { name: 'bn4f_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 132, col: 0 },
  { name: 'add_13', className: 'Add', row: 133, col: 1 },
  { name: 'activation_40', className: 'Activation', details: 'ReLU', row: 134, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // conv block 5a
  {
    name: 'res5a_branch2a',
    className: 'Conv2D',
    details: '512 1x1 filters, 2x2 strides, padding valid',
    row: 135,
    col: 0
  },
  { name: 'bn5a_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 136, col: 0 },
  { name: 'activation_41', className: 'Activation', details: 'ReLU', row: 137, col: 0 },
  {
    name: 'res5a_branch2b',
    className: 'Conv2D',
    details: '512 3x3 filters, 1x1 strides, padding same',
    row: 138,
    col: 0
  },
  { name: 'bn5a_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 139, col: 0 },
  { name: 'activation_42', className: 'Activation', details: 'ReLU', row: 140, col: 0 },
  {
    name: 'res5a_branch2c',
    className: 'Conv2D',
    details: '2048 1x1 filters, 1x1 strides, padding valid',
    row: 141,
    col: 0
  },
  {
    name: 'res5a_branch1',
    className: 'Conv2D',
    details: '2048 1x1 filters, 2x2 strides, padding valid',
    row: 135,
    col: 1
  },
  { name: 'bn5a_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 142, col: 0 },
  { name: 'bn5a_branch1', className: 'BatchNormalization', details: 'channel axis features', row: 136, col: 1 },
  { name: 'add_14', className: 'Add', row: 143, col: 1 },
  { name: 'activation_43', className: 'Activation', details: 'ReLU', row: 144, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 5b
  {
    name: 'res5b_branch2a',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 145,
    col: 0
  },
  { name: 'bn5b_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 146, col: 0 },
  { name: 'activation_44', className: 'Activation', details: 'ReLU', row: 147, col: 0 },
  {
    name: 'res5b_branch2b',
    className: 'Conv2D',
    details: '512 3x3 filters, 1x1 strides, padding same',
    row: 148,
    col: 0
  },
  { name: 'bn5b_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 149, col: 0 },
  { name: 'activation_45', className: 'Activation', details: 'ReLU', row: 150, col: 0 },
  {
    name: 'res5b_branch2c',
    className: 'Conv2D',
    details: '2048 1x1 filters, 1x1 strides, padding valid',
    row: 151,
    col: 0
  },
  { name: 'bn5b_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 152, col: 0 },
  { name: 'add_15', className: 'Add', row: 153, col: 1 },
  { name: 'activation_46', className: 'Activation', details: 'ReLU', row: 154, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // identity block 5c
  {
    name: 'res5c_branch2a',
    className: 'Conv2D',
    details: '512 1x1 filters, 1x1 strides, padding valid',
    row: 155,
    col: 0
  },
  { name: 'bn5c_branch2a', className: 'BatchNormalization', details: 'channel axis features', row: 156, col: 0 },
  { name: 'activation_47', className: 'Activation', details: 'ReLU', row: 157, col: 0 },
  {
    name: 'res5c_branch2b',
    className: 'Conv2D',
    details: '512 3x3 filters, 1x1 strides, padding same',
    row: 158,
    col: 0
  },
  { name: 'bn5c_branch2b', className: 'BatchNormalization', details: 'channel axis features', row: 159, col: 0 },
  { name: 'activation_48', className: 'Activation', details: 'ReLU', row: 160, col: 0 },
  {
    name: 'res5c_branch2c',
    className: 'Conv2D',
    details: '2048 1x1 filters, 1x1 strides, padding valid',
    row: 161,
    col: 0
  },
  { name: 'bn5c_branch2c', className: 'BatchNormalization', details: 'channel axis features', row: 162, col: 0 },
  { name: 'add_16', className: 'Add', row: 163, col: 1 },
  { name: 'activation_49', className: 'Activation', details: 'ReLU', row: 164, col: 1 },
  // /////////////////////////////////////////////////////////////////////
  // final
  {
    name: 'avg_pool',
    className: 'AveragePooling2D',
    details: '7x7 pool size, 7x7 strides, padding valid',
    row: 165,
    col: 1
  },
  { name: 'flatten_1', className: 'Flatten', details: '', row: 166, col: 1 },
  { name: 'fc1000', className: 'Dense', details: 'output dimensions 1000, softmax activation', row: 167, col: 1 }
]

export const ARCHITECTURE_CONNECTIONS = [
  // main
  { from: 'res2a_branch1', to: 'fc1000' },
  // initial + conv block 2a
  { from: 'zero_padding2d_1', to: 'bn2a_branch2c' },
  // identity block 2b
  { from: 'res2b_branch2a', to: 'bn2b_branch2c' },
  // identity block 2c
  { from: 'res2c_branch2a', to: 'bn2c_branch2c' },
  // conv block 3a
  { from: 'res3a_branch2a', to: 'bn3a_branch2c' },
  // identity block 3b
  { from: 'res3b_branch2a', to: 'bn3b_branch2c' },
  // identity block 3c
  { from: 'res3c_branch2a', to: 'bn3c_branch2c' },
  // identity block 3d
  { from: 'res3d_branch2a', to: 'bn3d_branch2c' },
  // conv block 4a
  { from: 'res4a_branch2a', to: 'bn4a_branch2c' },
  // identity block 4b
  { from: 'res4b_branch2a', to: 'bn4b_branch2c' },
  // identity block 4c
  { from: 'res4c_branch2a', to: 'bn4c_branch2c' },
  // identity block 4d
  { from: 'res4d_branch2a', to: 'bn4d_branch2c' },
  // identity block 4e
  { from: 'res4e_branch2a', to: 'bn4e_branch2c' },
  // identity block 4f
  { from: 'res4f_branch2a', to: 'bn4f_branch2c' },
  // conv block 5a
  { from: 'res5a_branch2a', to: 'bn5a_branch2c' },
  // identity block 5b
  { from: 'res5b_branch2a', to: 'bn5b_branch2c' },
  // identity block 5c
  { from: 'res5c_branch2a', to: 'bn5c_branch2c' },
  // block connections start
  { from: 'max_pooling2d_1', to: 'res2a_branch1', corner: 'top-right' },
  { from: 'activation_4', to: 'res2b_branch2a', corner: 'top-left' },
  { from: 'activation_7', to: 'res2c_branch2a', corner: 'top-left' },
  { from: 'activation_10', to: 'res3a_branch2a', corner: 'top-left' },
  { from: 'activation_13', to: 'res3b_branch2a', corner: 'top-left' },
  { from: 'activation_16', to: 'res3c_branch2a', corner: 'top-left' },
  { from: 'activation_19', to: 'res3d_branch2a', corner: 'top-left' },
  { from: 'activation_22', to: 'res4a_branch2a', corner: 'top-left' },
  { from: 'activation_25', to: 'res4b_branch2a', corner: 'top-left' },
  { from: 'activation_28', to: 'res4c_branch2a', corner: 'top-left' },
  { from: 'activation_31', to: 'res4d_branch2a', corner: 'top-left' },
  { from: 'activation_34', to: 'res4e_branch2a', corner: 'top-left' },
  { from: 'activation_37', to: 'res4f_branch2a', corner: 'top-left' },
  { from: 'activation_40', to: 'res5a_branch2a', corner: 'top-left' },
  { from: 'activation_43', to: 'res5b_branch2a', corner: 'top-left' },
  { from: 'activation_46', to: 'res5c_branch2a', corner: 'top-left' },
  // block connections to merge
  { from: 'bn2a_branch2c', to: 'add_1', corner: 'bottom-left' },
  { from: 'bn2b_branch2c', to: 'add_2', corner: 'bottom-left' },
  { from: 'bn2c_branch2c', to: 'add_3', corner: 'bottom-left' },
  { from: 'bn3a_branch2c', to: 'add_4', corner: 'bottom-left' },
  { from: 'bn3b_branch2c', to: 'add_5', corner: 'bottom-left' },
  { from: 'bn3c_branch2c', to: 'add_6', corner: 'bottom-left' },
  { from: 'bn3d_branch2c', to: 'add_7', corner: 'bottom-left' },
  { from: 'bn4a_branch2c', to: 'add_8', corner: 'bottom-left' },
  { from: 'bn4b_branch2c', to: 'add_9', corner: 'bottom-left' },
  { from: 'bn4c_branch2c', to: 'add_10', corner: 'bottom-left' },
  { from: 'bn4d_branch2c', to: 'add_11', corner: 'bottom-left' },
  { from: 'bn4e_branch2c', to: 'add_12', corner: 'bottom-left' },
  { from: 'bn4f_branch2c', to: 'add_13', corner: 'bottom-left' },
  { from: 'bn5a_branch2c', to: 'add_14', corner: 'bottom-left' },
  { from: 'bn5b_branch2c', to: 'add_15', corner: 'bottom-left' },
  { from: 'bn5c_branch2c', to: 'add_16', corner: 'bottom-left' }
]
