// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'core.Dropout.0': {
      weights: [
        { data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] },
        { data: [0.5, 0.7], shape: [2] }
      ],
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] },
      expected: { data: [7.3, -0.21], shape: [2] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
