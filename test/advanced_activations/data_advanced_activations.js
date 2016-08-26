// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

(function () {
  var DATA = {
    'advanced_activations.LeakyReLU.0': {
      input: { data: [0, 0.2, -0.5, -0.1, 1, 2], shape: [6] },
      expected: { data: [0.0, 0.2, -0.2, -0.04, 1.0, 2.0], shape: [6] }
    },
    'advanced_activations.PReLU.0': {
      input: { data: [0, 0.2, -0.5, -0.1, 1, 2], shape: [6] },
      weights: [
        { data: [-0.03, -0.02, 0.02, -0.03, -0.03, -0.01], shape: [6] }
      ],
      expected: { data: [0.0, 0.2, -0.01, 0.003, 1.0, 2.0], shape: [6] }
    },
    'advanced_activations.ELU.0': {
      input: { data: [0, 0.2, -0.5, -0.1, 1, 2], shape: [6] },
      expected: { data: [0.0, 0.2, -0.432816, -0.104679, 1.0, 2.0], shape: [6] }
    },
    'advanced_activations.ParametricSoftplus.0': {
      input: { data: [0, 0.2, -0.5, -0.1, 1, 2], shape: [6] },
      weights: [
        { data: [0.13, -0.02, 0.02, -0.03, -0.03, -0.01], shape: [6] },
        { data: [-0.03, -0.1, 0.02, 0.5, 0.2, 0.0], shape: [6] }
      ],
      expected: { data: [0.090109, -0.013664, 0.013763, -0.020054, -0.023944, -0.006931], shape: [6] }
    },
    'advanced_activations.ThresholdedReLU.0': {
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] },
      expected: { data: [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], shape: [6] }
    },
    'advanced_activations.SReLU.0': {
      input: { data: [0, 0.2, -0.5, -0.1, 1, 2], shape: [6] },
      weights: [
        { data: [0.13, -0.02, 0.02, -0.03, -0.03, -0.01], shape: [6] },
        { data: [-0.03, -0.1, 0.02, 0.5, 0.2, 0.0], shape: [6] },
        { data: [-0.9, 0.8, 0.0, -1.0, 0.7, 0.4], shape: [6] },
        { data: [0.1, 0.2, 0.3, 0.0, 0.5, -0.2], shape: [6] }
      ],
      expected: { data: [0.1339, 0.2, 0.0096, -0.065, 0.835, 0.068], shape: [6] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
