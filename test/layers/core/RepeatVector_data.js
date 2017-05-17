// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'core.RepeatVector.0': {
      expected: {
        data: [
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0,
          0.0,
          0.2,
          0.5,
          -0.1,
          1.0,
          2.0
        ],
        shape: [7, 6]
      },
      input: { data: [0, 0.2, 0.5, -0.1, 1, 2], shape: [6] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
