describe('convolutional layer: Cropping3D', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  before(function() {
    console.log('\n%cconvolutional layer: Cropping3D', styles.h1);
  });

  it(`[convolutional.Cropping3D.0] cropping ((1,1), (1,1), (1,1)) on 3x5x3x3 input, dim_ordering=tf`, function() {
    const key = `convolutional.Cropping3D.0`;
    console.log(`\n%c[${key}] cropping ((1,1), (1,1), (1,1)) on 3x5x3x3 input, dim_ordering=tf`, styles.h3);
    let testLayer = new layers.Cropping3D({ cropping: [ [ 1, 1 ], [ 1, 1 ], [ 1, 1 ] ], dimOrdering: 'tf' });
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

  it(`[convolutional.Cropping3D.0] cropping ((1,1), (1,1), (1,1)) on 3x5x3x3 input, dim_ordering=th`, function() {
    const key = `convolutional.Cropping3D.1`;
    console.log(`\n%c[${key}] cropping ((1,1), (1,1), (1,1)) on 3x5x3x3 input, dim_ordering=th`, styles.h3);
    let testLayer = new layers.Cropping3D({ cropping: [ [ 1, 1 ], [ 1, 1 ], [ 1, 1 ] ], dimOrdering: 'th' });
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

  it(`[convolutional.Cropping3D.2] cropping (3,2), (2,1), (2,3)) on 7x6x6x6 input, dim_ordering=tf`, function() {
    const key = `convolutional.Cropping3D.2`;
    console.log(`\n%c[${key}] cropping (3,2), (2,1), (2,3)) on 7x6x6x6 input, dim_ordering=tf`, styles.h3);
    let testLayer = new layers.Cropping3D({ cropping: [ [ 3, 2 ], [ 2, 1 ], [ 2, 3 ] ], dimOrdering: 'tf' });
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

  it(`[convolutional.Cropping3D.2] cropping (3,2), (2,1), (2,3)) on 7x6x6x6 input, dim_ordering=th`, function() {
    const key = `convolutional.Cropping3D.3`;
    console.log(`\n%c[${key}] cropping (3,2), (2,1), (2,3)) on 7x6x6x6 input, dim_ordering=th`, styles.h3);
    let testLayer = new layers.Cropping3D({ cropping: [ [ 3, 2 ], [ 2, 1 ], [ 2, 3 ] ], dimOrdering: 'th' });
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
