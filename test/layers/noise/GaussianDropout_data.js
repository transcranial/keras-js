// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'noise.GaussianDropout.0': {
      weights: [
        {
          shape: [6, 2],
          data: [
            -0.700122,
            -0.878386,
            1.273181,
            -1.531156,
            1.363569,
            0.900524,
            -1.798084,
            0.86898,
            0.627284,
            1.256685,
            -1.039475,
            0.014747
          ]
        },
        { shape: [2], data: [0.95711, -1.109336] }
      ],
      expected: { shape: [2], data: [0.621673, 0.233976] },
      input: { shape: [6], data: [0, 0.2, 0.5, -0.1, 1, 2] }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
