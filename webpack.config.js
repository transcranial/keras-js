const path = require('path')
const webpack = require('webpack')

const config = {
  entry: path.resolve(__dirname, 'src/index'),
  resolve: { extensions: ['.js'] },
  output: { path: path.resolve(__dirname, 'dist'), filename: 'keras.js', library: 'KerasJS', libraryTarget: 'umd' },
  module: {
    rules: [
      { test: /\.js$/, loader: 'babel-loader', exclude: /node_modules/ },
      { test: /\.(glsl|frag|vert)$/, use: ['raw-loader', 'glslify-loader'], exclude: /node_modules/ }
    ]
  },
  node: {
    fs: 'empty'
  }
}

if (process.env.NODE_ENV === 'production') {
  config.devtool = 'cheap-module-source-map'
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    //NOTE: possible bug in uglify-js: unused needs to be set to false or else library will not work properly
    new webpack.optimize.UglifyJsPlugin({ compress: { warnings: false, unused: false } })
  ]
} else {
  config.devtool = 'eval'
  config.plugins = [new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') })]
}

module.exports = config
