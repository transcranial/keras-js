import Vue from 'vue'
import VueMdl from 'vue-mdl'
import App from './App'
import router from './router'

Vue.use(VueMdl)

const app = new Vue(Object.assign({ router }, App))

app.$mount('#root')
