describe('convolutional layer: Deconvolution2D', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  const testParams = [
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 4, 3, 3 ],
      outputShape: [ 6, 6, 4 ],
      attrs: { activation: 'linear', borderMode: 'valid', subsample: [ 1, 1 ], dimOrdering: 'tf', bias: false }
    },
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 4, 3, 3 ],
      outputShape: [ 6, 6, 4 ],
      attrs: { activation: 'linear', borderMode: 'valid', subsample: [ 1, 1 ], dimOrdering: 'tf', bias: true }
    },
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 4, 3, 3 ],
      outputShape: [ 9, 9, 4 ],
      attrs: { activation: 'relu', borderMode: 'valid', subsample: [ 2, 2 ], dimOrdering: 'tf', bias: true }
    },
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 4, 3, 3 ],
      outputShape: [ 4, 4, 4 ],
      attrs: { activation: 'relu', borderMode: 'same', subsample: [ 1, 1 ], dimOrdering: 'tf', bias: true }
    },
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 5, 3, 3 ],
      outputShape: [ 8, 8, 5 ],
      attrs: { activation: 'relu', borderMode: 'same', subsample: [ 2, 2 ], dimOrdering: 'tf', bias: true }
    },
    {
      inputShape: [ 4, 4, 2 ],
      kernelShape: [ 5, 3, 3 ],
      outputShape: [ 7, 7, 5 ],
      attrs: { activation: 'relu', borderMode: 'same', subsample: [ 2, 2 ], dimOrdering: 'tf', bias: true }
    }
  ];

  before(function() {
    console.log('\n%cconvolutional layer: Deconvolution2D', styles.h1);
  });

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2);
    });

    testParams.forEach(({ inputShape, kernelShape, outputShape, attrs }, i) => {
      const key = `convolutional.Deconvolution2D.${i}`;
      const [ inputRows, inputCols, inputChannels ] = inputShape;
      const [ nbFilter, nbRow, nbCol ] = kernelShape;
      const [ outputRows, outputCols, outputChannels ] = outputShape;
      const title = `[${key}] [CPU] test: ${nbFilter} ${nbRow}x${nbCol} filters on ${inputRows}x${inputCols}x${inputChannels} input, ${outputRows}x${outputCols}x${outputChannels} output, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsample=${attrs.subsample}, dim_ordering='${attrs.dimOrdering}', bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Deconvolution2D(Object.assign({ nbFilter, nbRow, nbCol, outputShape }, attrs));
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)));
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor));
        const startTime = performance.now();
        t = testLayer.call(t);
        const endTime = performance.now();
        console.log('%cout', styles.h4, stringifyCondensed(t.tensor));
        logTime(startTime, endTime);
        const dataExpected = new Float32Array(TEST_DATA[key].expected.data);
        const shapeExpected = TEST_DATA[key].expected.shape;
        assert.deepEqual(t.tensor.shape, shapeExpected);
        assert.isTrue(approxEquals(t.tensor, dataExpected));
      });
    });
  });

  /*********************************************************
  * GPU
  *********************************************************/
  describe('GPU', function() {
    before(function() {
      console.log('\n%cGPU', styles.h2);
    });

    testParams.forEach(({ inputShape, kernelShape, outputShape, attrs }, i) => {
      const key = `convolutional.Deconvolution2D.${i}`;
      const [ inputRows, inputCols, inputChannels ] = inputShape;
      const [ nbFilter, nbRow, nbCol ] = kernelShape;
      const [ outputRows, outputCols, outputChannels ] = outputShape;
      const title = `[${key}] [GPU] test: ${nbFilter} ${nbRow}x${nbCol} filters on ${inputRows}x${inputCols}x${inputChannels} input, ${outputRows}x${outputCols}x${outputChannels} output, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsample=${attrs.subsample}, dim_ordering='${attrs.dimOrdering}', bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Deconvolution2D(
          Object.assign({ nbFilter, nbRow, nbCol, outputShape }, attrs, { gpu: true })
        );
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)));
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor));
        const startTime = performance.now();
        t = testLayer.call(t);
        const endTime = performance.now();
        console.log('%cout', styles.h4, stringifyCondensed(t.tensor));
        logTime(startTime, endTime);
        const dataExpected = new Float32Array(TEST_DATA[key].expected.data);
        const shapeExpected = TEST_DATA[key].expected.shape;
        assert.deepEqual(t.tensor.shape, shapeExpected);
        assert.isTrue(approxEquals(t.tensor, dataExpected));
      });
    });
  });
});
