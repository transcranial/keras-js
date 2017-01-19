describe('convolutional layer: Convolution1D', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  const testParams = [
    {
      inputShape: [ 5, 2 ],
      kernelShape: [ 4, 3 ],
      attrs: { activation: 'linear', borderMode: 'valid', subsampleLength: 1, bias: true }
    },
    {
      inputShape: [ 6, 3 ],
      kernelShape: [ 4, 3 ],
      attrs: { activation: 'linear', borderMode: 'valid', subsampleLength: 1, bias: false }
    },
    {
      inputShape: [ 4, 6 ],
      kernelShape: [ 2, 3 ],
      attrs: { activation: 'sigmoid', borderMode: 'same', subsampleLength: 2, bias: true }
    },
    {
      inputShape: [ 8, 3 ],
      kernelShape: [ 2, 7 ],
      attrs: { activation: 'tanh', borderMode: 'same', subsampleLength: 1, bias: true }
    }
  ];

  before(function() {
    console.log('\n%cconvolutional layer: Convolution1D', styles.h1);
  });

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2);
    });

    testParams.forEach(({ inputShape, kernelShape, attrs }, i) => {
      const key = `convolutional.Convolution1D.${i}.legacy`;
      const [ inputLength, inputFeatures ] = inputShape;
      const [ nbFilter, filterLength ] = kernelShape;
      const title = `[${key}] [CPU] test: ${nbFilter} length ${filterLength} filters on ${inputLength}x${inputFeatures} input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsampleLength=${attrs.subsampleLength}, bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Convolution1D(Object.assign({ nbFilter, filterLength }, attrs));
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

    testParams.forEach(({ inputShape, kernelShape, attrs }, i) => {
      const key = `convolutional.Convolution1D.${i}`;
      const [ inputLength, inputFeatures ] = inputShape;
      const [ nbFilter, filterLength ] = kernelShape;
      const title = `[${key}] [CPU] test: ${nbFilter} length ${filterLength} filters on ${inputLength}x${inputFeatures} input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsampleLength=${attrs.subsampleLength}, bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Convolution1D(Object.assign({ nbFilter, filterLength }, attrs));
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

    testParams.forEach(({ inputShape, kernelShape, attrs }, i) => {
      const key = `convolutional.Convolution1D.${i}.legacy`;
      const [ inputLength, inputFeatures ] = inputShape;
      const [ nbFilter, filterLength ] = kernelShape;
      const title = `[${key}] [GPU] test: ${nbFilter} length ${filterLength} filters on ${inputLength}x${inputFeatures} input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsampleLength=${attrs.subsampleLength}, bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Convolution1D(Object.assign({ nbFilter, filterLength }, attrs, { gpu: true }));
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

    testParams.forEach(({ inputShape, kernelShape, attrs }, i) => {
      const key = `convolutional.Convolution1D.${i}`;
      const [ inputLength, inputFeatures ] = inputShape;
      const [ nbFilter, filterLength ] = kernelShape;
      const title = `[${key}] [GPU] test: ${nbFilter} length ${filterLength} filters on ${inputLength}x${inputFeatures} input, activation='${attrs.activation}', border_mode='${attrs.borderMode}', subsampleLength=${attrs.subsampleLength}, bias=${attrs.bias}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.Convolution1D(Object.assign({ nbFilter, filterLength }, attrs, { gpu: true }));
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
