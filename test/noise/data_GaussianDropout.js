// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

(function () {
  var DATA = {
    'core.GaussianDropout.0': {
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] },
      weights: [
        { data: [-0.7, -0.9, 1.3, -1.5, 1.4, 0.9, -1.8, 0.9, 0.6, 1.3, -1.0, 0.01], shape: [6, 2] },
        { data: [1.0, -1.1], shape: [2] }
      ],
      expected: { data: [6.2, 0.23], shape: [2] }
    },
    'core.GaussianNoise.0': {
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] },
      weights: [
        { data: [-0.7, -0.9, 1.3, -1.5, 1.4, 0.9, -1.8, 0.9, 0.6, 1.3, -1.0, 0.01], shape: [6, 2] },
        { data: [1.0, -1.1], shape: [2] }
      ],
      expected: { data: [6.2, 0.23], shape: [2] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
