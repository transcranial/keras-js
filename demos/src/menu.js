/* global Vue */
import './menu.css'

export const Menu = Vue.extend({
  props: ['currentView'],
  template: require('raw-loader!./menu.template.html')
})
