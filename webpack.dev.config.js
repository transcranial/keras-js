const path = require('path')
const webpack = require('webpack')

module.exports = {
  entry: [
    'babel-polyfill',
    path.join(__dirname, 'src/index')
  ],
  output: {
    path: path.join(__dirname, 'dist'),
    filename: 'keras.js',
    library: 'KerasJS',
    libraryTarget: 'umd'
  },
  devtool: 'eval',
  module: {
    loaders: [
      {
        test: /\.js$/,
        loaders: ['babel-loader'],
        exclude: /node_modules/
      },
      {
        test: /\.(glsl|frag|vert)$/,
        loaders: ['raw-loader', 'glslify-loader']
      }
    ]
  },
  resolve: {
    extensions: ['.js']
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify('development')
    })
  ],
  performance: {
    hints: false
  }
}
