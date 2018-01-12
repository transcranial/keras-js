<template>
  <div id="app">
    <v-app>
      <v-navigation-drawer v-model="showNav" absolute fixed floating app>
        <main-menu :currentView="currentView"></main-menu>
      </v-navigation-drawer>
      <v-toolbar app dark color="primary">
        <v-toolbar-side-icon @click.stop="showNav = !showNav"></v-toolbar-side-icon>
        <v-toolbar-title>{{ currentTitle }}</v-toolbar-title>
        <v-btn v-if="currentTitle" icon @click.stop="showInfoPanel = true">
          <v-icon>fa-info-circle</v-icon>
        </v-btn>
        <v-spacer></v-spacer>
        <v-btn icon href="https://github.com/transcranial/keras-js">
          <v-icon>fa-github</v-icon>
        </v-btn>
      </v-toolbar>
      <v-content>
        <v-container fluid>
          <router-view :hasWebGL="hasWebGL"></router-view>
        </v-container>
      </v-content>
      <v-footer app></v-footer>
      <info-panel 
        :showInfoPanel="showInfoPanel" 
        :currentView="currentView" 
        :close="closeInfoPanel"
      ></info-panel>
    </v-app>
  </div>
</template>

<script>
import MainMenu from './components/MainMenu'
import InfoPanel from './components/InfoPanel'
import { DEMO_TITLES } from './data/demo-titles'

export default {
  components: { MainMenu, InfoPanel },
  data() {
    return {
      showNav: true,
      showInfoPanel: false,
      hasWebGL: KerasJS.GPU_SUPPORT
    }
  },
  computed: {
    currentView() {
      const path = this.$route.path
      return path.replace(/^\//, '') || 'home'
    },
    currentTitle() {
      return DEMO_TITLES[this.currentView]
    }
  },
  methods: {
    closeInfoPanel() {
      this.showInfoPanel = false
    }
  }
}
</script>

<style lang="postcss">
@import 'https://fonts.googleapis.com/css?family=Open+Sans:300, 400, 600, 700';
@import 'https://fonts.googleapis.com/css?family=Share+Tech+Mono';
@import 'https://fonts.googleapis.com/css?family=Nothing+You+Could+Do';

@import './variables.css';

.application {
  font-family: var(--font-sans-serif);
}

.application.theme--light {
  background: linear-gradient(0deg, #cccccc, #f0f0f0) !important;
  color: var(--color-darkgray);
}

footer {
  background: #cccccc !important;
}

a {
  text-decoration: none;
}

.demo {
  position: relative;
}

/*******************************************************************/
/* Vuetify overrides */

.navigation-drawer {
  background-color: whitesmoke !important;
}

.input-group--select .input-group__selections__comma,
.input-group input,
.input-group textarea {
  font-size: 14px !important;
  color: var(--color-darkgray) !important;
}

.input-group:not(.input-group--error) label {
  font-size: 14px !important;
  color: var(--color-lightgray) !important;
}

.list .list__tile:not(.list__tile--active) {
  color: var(--color-darkgray) !important;
}

.list__tile {
  font-size: 14px !important;
  height: 35px !important;
  font-family: var(--font-monospace);
}
</style>
