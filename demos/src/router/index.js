import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../components/Home'
import MnistCnn from '../components/models/MnistCnn'

Vue.use(VueRouter)

const router = new VueRouter({
  routes: [{ path: '/', component: Home }, { path: '/mnist-cnn', component: MnistCnn }]
})

export default router
