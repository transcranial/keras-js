/* global Vue */

import { Menu } from './menu'
import { Home } from './home'
import { MnistCnn } from './mnist-cnn'

Vue.component('menu', Menu)
Vue.component('home', Home)
Vue.component('mnist-cnn', MnistCnn)

const app = new Vue({
  el: '#app',
  data: {
    currentView: 'home'
  }
})

// Simple routing

function matchRoute () {
  const routes = ['mnist-cnn']

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
