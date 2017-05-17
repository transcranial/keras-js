describe('embeddings layer: Embedding', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { inputShape: [7], attrs: { inputDim: 5, outputDim: 3, inputLength: 7, maskZero: false, dropout: 0 } },
    { inputShape: [10], attrs: { inputDim: 20, outputDim: 5, inputLength: 10, maskZero: true, dropout: 0 } },
    { inputShape: [5], attrs: { inputDim: 33, outputDim: 2, inputLength: 5, maskZero: false, dropout: 0.5 } }
  ]

  before(function() {
    console.log('\n%cembeddings layer: Embedding', styles.h1)
  })

  testParams.forEach(({ attrs }, i) => {
    const key = `embeddings.Embedding.${i}`
    const title = `[${key}] test: inputDim='${attrs.inputDim}', outputDim=${attrs.outputDim}, inputLength=${attrs.inputLength}, maskZero=${attrs.maskZero}, dropout=${attrs.dropout}`

    it(title, function() {
      console.log(`\n%c${title}`, styles.h3)
      let testLayer = new layers.Embedding(attrs)
      testLayer.setWeights(TEST_DATA[key].weights.map(w => new KerasJS.Tensor(w.data, w.shape)))
      let t = new KerasJS.Tensor(TEST_DATA[key].input.data, TEST_DATA[key].input.shape)
      console.log('%cin', styles.h4, stringifyCondensed(t.tensor))
      const startTime = performance.now()
      t = testLayer.call(t)
      const endTime = performance.now()
      console.log('%cout', styles.h4, stringifyCondensed(t.tensor))
      logTime(startTime, endTime)
      const dataExpected = new Float32Array(TEST_DATA[key].expected.data)
      const shapeExpected = TEST_DATA[key].expected.shape
      assert.deepEqual(t.tensor.shape, shapeExpected)
      assert.isTrue(approxEquals(t.tensor, dataExpected))
    })
  })
})
