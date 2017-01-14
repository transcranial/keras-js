describe('core layer: Highway', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  before(function() {
    console.log('\n%ccore layer: Highway', styles.h1);
  });

  it(
    '[core.Highway.0] should produce expected values, transformBias=-2, activation=linear bias=true',
    function() {
      const key = 'core.Highway.0';
      console.log(
        `\n%c[${key}] transformBias=-2, activation=linear bias=true`,
        styles.h3
      );
      let testLayer = new layers.Highway({
        transformBias: -2,
        activation: 'linear',
        bias: true
      });
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
    '[core.Highway.1] should produce expected values, transformBias=-5, activation=tanh bias=true',
    function() {
      const key = 'core.Highway.1';
      console.log(
        `\n%c[${key}] transformBias=-5, activation=tanh bias=true`,
        styles.h3
      );
      let testLayer = new layers.Highway({
        transformBias: -5,
        activation: 'tanh',
        bias: true
      });
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
    '[core.Highway.2] should produce expected values, transformBias=1, activation=hardSigmoid bias=false',
    function() {
      const key = 'core.Highway.2';
      console.log(
        `\n%c[${key}] transformBias=1, activation=hardSigmoid bias=false`,
        styles.h3
      );
      let testLayer = new layers.Highway({
        transformBias: 1,
        activation: 'hardSigmoid',
        bias: false
      });
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
