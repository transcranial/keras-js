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
  devtool: 'cheap-module-source-map',
  module: {
    loaders: [
      {
        test: /\.js$/,
        loaders: ['babel-loader'],
        exclude: /node_modules/
      },
      {
        test: /\.(glsl|frag|vert)$/,
        loaders: ['raw-loader', 'glslify-loader'],
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: ['.js']
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: JSON.stringify('production')
      }
    }),
    new webpack.optimize.UglifyJsPlugin({
      compress: { screw_ie8: true, warnings: false },
      mangle: { screw_ie8: true },
      output: { comments: false, screw_ie8: true }
    })
  ],
  performance: {
    hints: false
  }
}
