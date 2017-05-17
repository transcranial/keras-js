// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'core.Dense.0': {
      weights: [
        { data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] },
        { data: [0.5, 0.7], shape: [2] }
      ],
      expected: { data: [7.3, -0.21], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    },
    'core.Dense.1': {
      weights: [
        { data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] },
        { data: [0.5, 0.7], shape: [2] }
      ],
      expected: { data: [0.999325, 0.447692], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    },
    'core.Dense.2': {
      weights: [{ data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] }],
      expected: { data: [6.801113, 0.338274], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    },
    'core.Dense.3': {
      weights: [
        { data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] },
        { data: [0.5, 0.7], shape: [2] }
      ],
      expected: { data: [7.3, -0.21], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    },
    'core.Dense.4': {
      weights: [
        { data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] },
        { data: [0.5, 0.7], shape: [2] }
      ],
      expected: { data: [0.999325, 0.447692], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    },
    'core.Dense.5': {
      weights: [{ data: [0.1, 0.4, 0.5, 0.1, 1.0, -2.0, 0.0, 0.3, 0.2, 0.1, 3.0, 0.0], shape: [6, 2] }],
      expected: { data: [6.801113, 0.338274], shape: [2] },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
