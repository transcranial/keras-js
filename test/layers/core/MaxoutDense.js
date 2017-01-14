describe('core layer: MaxoutDense', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  before(function() {
    console.log('\n%ccore layer: MaxoutDense', styles.h1);
  });

  it(
    '[core.MaxoutDense.0] should produce expected values, nbFeature=4, bias=true',
    function() {
      const key = 'core.MaxoutDense.0';
      console.log(`\n%c[${key}] nbFeature=4, bias=true`, styles.h3);
      let testLayer = new layers.MaxoutDense({ outputDim: 3 });
      testLayer.setWeights(
        TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape))
      );
      let t = new KerasJS.Tensor(
        TEST_DATA[key].input.data,
        TEST_DATA[key].input.shape
      );
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
    }
  );

  it(
    '[core.MaxoutDense.1] should produce expected values, nbFeature=7, bias=false',
    function() {
      const key = 'core.MaxoutDense.1';
      console.log(`\n%c[${key}] nbFeature=7, bias=false`, styles.h3);
      let testLayer = new layers.MaxoutDense({ outputDim: 3, bias: false });
      testLayer.setWeights(
        TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape))
      );
      let t = new KerasJS.Tensor(
        TEST_DATA[key].input.data,
        TEST_DATA[key].input.shape
      );
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
    }
  );
});
