/* global Vue, VueMdl */

import './index.css'

import { Menu } from './menu'
import { Home } from './home'
import { MnistCnn } from './mnist-cnn'
import { MnistVae } from './mnist-vae'
import { ResNet50 } from './resnet50'

Vue.component('menu', Menu)
Vue.component('home', Home)
Vue.component('mnist-cnn', MnistCnn)
Vue.component('mnist-vae', MnistVae)
Vue.component('resnet50', ResNet50)

Vue.use(VueMdl.default)

const app = new Vue({
  el: '#app',
  data: {
    currentView: 'home'
  }
})

// Simple routing

function matchRoute () {
  const routes = [
    'mnist-cnn',
    'mnist-vae',
    'resnet50'
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
