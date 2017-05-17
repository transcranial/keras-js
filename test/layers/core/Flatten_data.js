// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'core.Flatten.0': {
      expected: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'core.Flatten.1': {
      expected: { shape: [6], data: [0.0, 0.2, 0.5, -0.1, 1.0, 2.0] },
      input: { shape: [3, 2], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    },
    'core.Flatten.2': {
      expected: { shape: [12], data: [0.0, 0.2, 0.5, -0.1, 1.0, 2.0, 0.0, 0.2, 0.5, -0.1, 1.0, 2.0] },
      input: { shape: [3, 2, 2], data: [0, 0.2, 0.5, -0.1, 1, 2, 0, 0.2, 0.5, -0.1, 1, 2] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
