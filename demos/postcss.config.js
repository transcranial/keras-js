module.exports = {
  plugins: [require('postcss-import')(), require('postcss-cssnext')({ browsers: ['>0.5%'] })]
}
