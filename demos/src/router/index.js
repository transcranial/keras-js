import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../components/Home'

Vue.use(VueRouter)

const router = new VueRouter({
  routes: [{ path: '/', component: Home }]
})

export default router
