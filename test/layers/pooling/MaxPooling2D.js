describe('pooling layer: MaxPooling2D', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  const testParams = [
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: null,
        borderMode: 'valid',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: [ 1, 1 ],
        borderMode: 'valid',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 7, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: [ 2, 1 ],
        borderMode: 'valid',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: null,
        borderMode: 'valid',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: [ 3, 3 ],
        borderMode: 'valid',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: null,
        borderMode: 'same',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: [ 1, 1 ],
        borderMode: 'same',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 7, 3 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: [ 2, 1 ],
        borderMode: 'same',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: null,
        borderMode: 'same',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 6, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: [ 3, 3 ],
        borderMode: 'same',
        dimOrdering: 'tf'
      }
    },
    {
      inputShape: [ 5, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: [ 2, 2 ],
        borderMode: 'valid',
        dimOrdering: 'th'
      }
    },
    {
      inputShape: [ 5, 6, 3 ],
      attrs: {
        poolSize: [ 3, 3 ],
        strides: [ 1, 1 ],
        borderMode: 'same',
        dimOrdering: 'th'
      }
    },
    {
      inputShape: [ 4, 6, 4 ],
      attrs: {
        poolSize: [ 2, 2 ],
        strides: null,
        borderMode: 'valid',
        dimOrdering: 'th'
      }
    }
  ];

  before(function() {
    console.log('\n%cpooling layer: MaxPooling2D', styles.h1);
  });

  testParams.forEach(({ inputShape, attrs }, i) => {
    const key = `pooling.MaxPooling2D.${i}`;
    const [ inputRows, inputCols, inputChannels ] = inputShape;
    const title = `[${key}] test: ${inputRows}x${inputCols}x${inputChannels} input, poolSize='${attrs.poolSize}', strides=${attrs.strides}, borderMode=${attrs.borderMode}, dimOrdering=${attrs.dimOrdering}`;

    it(title, function() {
      console.log(`\n%c${title}`, styles.h3);
      let testLayer = new layers.MaxPooling2D(attrs);
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
    });
  });
});
