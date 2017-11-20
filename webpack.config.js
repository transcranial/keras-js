const path = require('path')
const webpack = require('webpack')

const config = {
  entry: path.resolve(__dirname, 'src/index'),
  resolve: { extensions: ['.js'] },
  output: { path: path.resolve(__dirname, 'dist'), filename: 'keras.min.js', library: 'KerasJS', libraryTarget: 'umd' },
  module: {
    rules: [{ test: /\.js$/, loader: 'babel-loader', exclude: /node_modules/ }]
  },
  node: {
    fs: 'empty'
  }
}

if (process.env.NODE_ENV === 'production') {
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    // scope hoisting
    new webpack.optimize.ModuleConcatenationPlugin(),
    // uglify: unused needs to be set to false or else library will not work properly
    new webpack.optimize.UglifyJsPlugin({
      compress: { warnings: false, unused: false },
      output: { comments: false }
    })
  ]
} else {
  config.devtool = 'eval'
  config.plugins = [new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') })]
}

module.exports = config
