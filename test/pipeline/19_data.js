// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    pipeline_19: {
      weights: [],
      input: {
        data: [
          0.30717917,
          -0.76998611,
          0.90056573,
          -0.0356172,
          0.74494907,
          -0.57533464,
          -0.91858075,
          -0.20561108,
          -0.53373561
        ],
        shape: [3, 3]
      },
      expected: { data: [0.30717918, 0.0, 0.90056574, 0.0, 0.74494904, 0.0, 0.0, 0.0, 0.0], shape: [3, 3] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
