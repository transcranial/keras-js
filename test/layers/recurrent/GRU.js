describe('recurrent layer: GRU', function() {
  const assert = chai.assert;
  const styles = testGlobals.styles;
  const logTime = testGlobals.logTime;
  const stringifyCondensed = testGlobals.stringifyCondensed;
  const approxEquals = KerasJS.testUtils.approxEquals;
  const layers = KerasJS.layers;

  const testParams = [
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: false,
        goBackwards: false,
        stateful: false
      }
    },
    {
      inputShape: [ 8, 5 ],
      attrs: {
        outputDim: 5,
        activation: 'sigmoid',
        innerActivation: 'sigmoid',
        returnSequences: false,
        goBackwards: false,
        stateful: false
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: true,
        goBackwards: false,
        stateful: false
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: false,
        goBackwards: true,
        stateful: false
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: true,
        goBackwards: true,
        stateful: false
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: false,
        goBackwards: false,
        stateful: true
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: true,
        goBackwards: false,
        stateful: true
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: false,
        goBackwards: true,
        stateful: true
      }
    },
    {
      inputShape: [ 3, 6 ],
      attrs: {
        outputDim: 4,
        activation: 'tanh',
        innerActivation: 'hardSigmoid',
        returnSequences: true,
        goBackwards: true,
        stateful: true
      }
    }
  ];

  before(function() {
    console.log('\n%crecurrent layer: GRU', styles.h1);
  });

  /*********************************************************
  * CPU
  *********************************************************/
  describe('CPU', function() {
    before(function() {
      console.log('\n%cCPU', styles.h2);
    });

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `recurrent.GRU.${i}`;
      const title = `[${key}] [CPU] test: ${inputShape[0]}x${inputShape[1]} input, activation='${attrs.activation}', innerActivation='${attrs.innerActivation}', returnSequences=${attrs.returnSequences}, goBackwards=${attrs.goBackwards}, stateful=${attrs.stateful}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.GRU(attrs);
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)));
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor));
        const startTime = performance.now();

        // To test statefulness, we run call() twice (see corresponding jupyter notebook)
        t = testLayer.call(t);
        if (attrs.stateful) {
          t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
          t = testLayer.call(t);
        }

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

    testParams.forEach(({ inputShape, attrs }, i) => {
      const key = `recurrent.GRU.${i}`;
      const title = `[${key}] [GPU] test: ${inputShape[0]}x${inputShape[1]} input, activation='${attrs.activation}', innerActivation='${attrs.innerActivation}', returnSequences=${attrs.returnSequences}, goBackwards=${attrs.goBackwards}, stateful=${attrs.stateful}`;

      it(title, function() {
        console.log(`\n%c${title}`, styles.h3);
        let testLayer = new layers.GRU(Object.assign(attrs, { gpu: true }));
        testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)));
        let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
        console.log('%cin', styles.h4, stringifyCondensed(t.tensor));
        const startTime = performance.now();

        // To test statefulness, we run call() twice (see corresponding jupyter notebook)
        t = testLayer.call(t);
        if (attrs.stateful) {
          t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape);
          t = testLayer.call(t);
        }

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
