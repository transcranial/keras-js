// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

(function() {
  var DATA = {
    'core.GaussianDropout.0': {
      input: { data: [ 0, 0.2, 0.5, -0.1, 1, 2 ], shape: [ 6 ] },
      weights: [
        {
          data: [
            -0.70012161,
            -0.8783863,
            1.27318097,
            -1.53115632,
            1.36356888,
            0.90052448,
            -1.7980838,
            0.86897991,
            0.62728353,
            1.25668481,
            -1.03947462,
            0.01474715
          ],
          shape: [ 6, 2 ]
        },
        { data: [ 0.95711006, -1.10933608 ], shape: [ 2 ] }
      ],
      expected: { data: [ 0.621673, 0.233976 ], shape: [ 2 ] }
    },
    'core.GaussianNoise.0': {
      input: { data: [ 0, 0.2, 0.5, -0.1, 1, 2 ], shape: [ 6 ] },
      weights: [
        {
          data: [
            -0.70012161,
            -0.8783863,
            1.27318097,
            -1.53115632,
            1.36356888,
            0.90052448,
            -1.7980838,
            0.86897991,
            0.62728353,
            1.25668481,
            -1.03947462,
            0.01474715
          ],
          shape: [ 6, 2 ]
        },
        { data: [ 0.95711006, -1.10933608 ], shape: [ 2 ] }
      ],
      expected: { data: [ 0.621673, 0.233976 ], shape: [ 2 ] }
    }
  };

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA);
})();
