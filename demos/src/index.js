/* global Vue, VueMdl, WebGLRenderingContext */

import './index.css'

import { Menu } from './menu'
import { Home } from './home'
import { MnistCnn } from './mnist-cnn'
import { MnistVae } from './mnist-vae'
import { ResNet50 } from './resnet50'
import { InceptionV3 } from './inception-v3'
import { ImdbBidirectionalLstm } from './imdb-bidirectional-lstm'

Vue.component('menu', Menu)
Vue.component('home', Home)
Vue.component('mnist-cnn', MnistCnn)
Vue.component('mnist-vae', MnistVae)
Vue.component('resnet50', ResNet50)
Vue.component('inception-v3', InceptionV3)
Vue.component('imdb-bidirectional-lstm', ImdbBidirectionalLstm)

Vue.use(VueMdl.default)

const app = new Vue({
  el: '#app',

  data: function () {
    return {
      currentView: 'home',
      hasWebgl: true
    }
  },

  created: function () {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
    // Report the result.
    if (gl && gl instanceof WebGLRenderingContext) {
      this.hasWebgl = true
    } else {
      this.hasWebgl = false
    }
  }
})

// Simple routing

function matchRoute () {
  const routes = [
    'mnist-cnn',
    'mnist-vae',
    'resnet50',
    'inception-v3',
    'imdb-bidirectional-lstm'
  ]

  const { hash } = window.location
  const route = hash.substr(2)
  if (routes.indexOf(route) > -1) {
    app.currentView = route
  } else {
    app.currentView = 'home'
  }
}

window.addEventListener('load', matchRoute)
window.addEventListener('hashchange', matchRoute)
