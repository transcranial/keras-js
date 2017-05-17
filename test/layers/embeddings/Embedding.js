describe('embeddings layer: Embedding', function() {
  const assert = chai.assert
  const styles = testGlobals.styles
  const logTime = testGlobals.logTime
  const stringifyCondensed = testGlobals.stringifyCondensed
  const approxEquals = KerasJS.testUtils.approxEquals
  const layers = KerasJS.layers

  const testParams = [
    { attrs: { input_dim: 5, output_dim: 3, input_length: 7, mask_zero: false } },
    { attrs: { input_dim: 20, output_dim: 5, input_length: 10, mask_zero: true } },
    { attrs: { input_dim: 33, output_dim: 2, input_length: 5, mask_zero: false } }
  ]

  before(function() {
    console.log('\n%cembeddings layer: Embedding', styles.h1)
  })

  testParams.forEach(({ attrs }, i) => {
    const key = `embeddings.Embedding.${i}`
    const title = `[${key}] test: input_dim='${attrs.input_dim}', output_dim=${attrs.output_dim}, input_length=${attrs.input_length}, mask_zero=${attrs.mask_zero}`

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
