const path = require('path')
const webpack = require('webpack')

const config = {
  entry: path.resolve(__dirname, 'src/index'),
  output: { path: path.resolve(__dirname, 'dist'), filename: 'keras.js', library: 'KerasJS', libraryTarget: 'umd' },
  devtool: 'eval',
  module: {
    rules: [
      { test: /\.js$/, use: ['babel-loader'], exclude: /node_modules/ },
      { test: /\.(glsl|frag|vert)$/, use: ['raw-loader', 'glslify-loader'], exclude: /node_modules/ }
    ]
  },
  node: {
    fs: 'empty'
  },
  plugins: []
}

// NODE_ENV defaults to 'development'
if (process.env.NODE_ENV === 'production') {
  config.devtool = 'cheap-module-source-map'
  config.plugins = config.plugins.concat([
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    new webpack.optimize.UglifyJsPlugin({
      sourceMap: true,
      compress: { screw_ie8: true, warnings: false },
      mangle: { screw_ie8: true },
      output: { comments: false, screw_ie8: true }
    })
  ])
} else {
  config.plugins = config.plugins.concat([
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') })
  ])
}

module.exports = config
