const path = require('path')
const webpack = require('webpack')

const config = {
  entry: [path.resolve(__dirname, 'src/index')],
  output: { path: path.resolve(__dirname, 'dist'), filename: 'bundle.js' },
  devtool: 'eval',
  module: {
    rules: [
      { enforce: 'pre', test: /\.vue$/, use: ['eslint-loader'], exclude: /node_modules/ },
      { enforce: 'pre', test: /\.js$/, use: ['eslint-loader'], exclude: /node_modules/ },
      { test: /\.vue$/, use: ['vue-loader'], exclude: /node_modules/ },
      { test: /\.js$/, use: ['babel-loader'], exclude: /node_modules/ }
    ]
  }
}

// NODE_ENV defaults to 'development'
if (process.env.NODE_ENV === 'production') {
  config.devtool = 'cheap-module-source-map'
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    new webpack.optimize.UglifyJsPlugin({
      sourceMap: true,
      compress: { screw_ie8: true, warnings: false },
      mangle: { screw_ie8: true },
      output: { comments: false, screw_ie8: true }
    })
  ]
} else {
  config.plugins = [new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') })]
}

module.exports = config
