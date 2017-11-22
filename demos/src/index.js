import '@babel/polyfill'
import Vue from 'vue'
import Vuetify from 'vuetify'
import App from './App'
import router from './router'
import 'vuetify/dist/vuetify.min.css'

Vue.use(Vuetify, {
  theme: {
    primary: '#1bbc9b',
    secondary: '#69707a',
    accent: '#f5d76e',
    error: '#d24d57'
  }
})

const app = new Vue(Object.assign({ router }, App))

app.$mount('#root')
