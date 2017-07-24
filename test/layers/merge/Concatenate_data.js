// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'merge.Concatenate.0': {
      weights: [
        { shape: [6, 2], data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0] },
        { shape: [2], data: [0.5, 0.7] },
        { shape: [6, 2], data: [1.0, 0.0, -0.9, 0.6, -0.7, 0.0, 0.2, 0.4, 0.0, 0.0, -1.0, 2.3] },
        { shape: [2], data: [0.1, -0.2] }
      ],
      expected: { shape: [4], data: [7.3, -0.21, -2.45, 4.48] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'merge.Concatenate.1': {
      weights: [
        { shape: [6, 2], data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0] },
        { shape: [2], data: [0.5, 0.7] },
        { shape: [6, 2], data: [1.0, 0.0, -0.9, 0.6, -0.7, 0.0, 0.2, 0.4, 0.0, 0.0, -1.0, 2.3] },
        { shape: [2], data: [0.1, -0.2] }
      ],
      expected: { shape: [3, 4], data: [7.3, -0.21, -2.45, 4.48, 7.3, -0.21, -2.45, 4.48, 7.3, -0.21, -2.45, 4.48] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'merge.Concatenate.2': {
      weights: [
        { shape: [6, 2], data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0] },
        { shape: [2], data: [0.5, 0.7] },
        { shape: [6, 2], data: [1.0, 0.0, -0.9, 0.6, -0.7, 0.0, 0.2, 0.4, 0.0, 0.0, -1.0, 2.3] },
        { shape: [2], data: [0.1, -0.2] }
      ],
      expected: { shape: [6, 2], data: [7.3, -0.21, 7.3, -0.21, 7.3, -0.21, -2.45, 4.48, -2.45, 4.48, -2.45, 4.48] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'merge.Concatenate.3': {
      weights: [
        { shape: [6, 2], data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0] },
        { shape: [2], data: [0.5, 0.7] },
        { shape: [6, 2], data: [1.0, 0.0, -0.9, 0.6, -0.7, 0.0, 0.2, 0.4, 0.0, 0.0, -1.0, 2.3] },
        { shape: [2], data: [0.1, -0.2] }
      ],
      expected: { shape: [6, 2], data: [7.3, -0.21, 7.3, -0.21, 7.3, -0.21, -2.45, 4.48, -2.45, 4.48, -2.45, 4.48] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'merge.Concatenate.4': {
      weights: [
        { shape: [6, 2], data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0] },
        { shape: [2], data: [0.5, 0.7] },
        { shape: [6, 2], data: [1.0, 0.0, -0.9, 0.6, -0.7, 0.0, 0.2, 0.4, 0.0, 0.0, -1.0, 2.3] },
        { shape: [2], data: [0.1, -0.2] }
      ],
      expected: { shape: [3, 4], data: [7.3, -0.21, -2.45, 4.48, 7.3, -0.21, -2.45, 4.48, 7.3, -0.21, -2.45, 4.48] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
