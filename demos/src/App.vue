<template>
  <div id="app">
    <div class="columns">
      <div class="column is-3" style="min-width: 350px; max-width: 450px;">
        <main-menu></main-menu>
      </div>
      <div class="column is-9">
        <router-view :has-webgl="hasWebgl"></router-view>
      </div>
    </div>
  </div>
</template>

<script>
import MainMenu from './components/MainMenu'

export default {
  components: { MainMenu },
  data: function() {
    return { hasWebgl: true }
  },
  created: function() {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
    // Report the result.
    if (gl && gl instanceof WebGLRenderingContext) {
      this.hasWebgl = true
    } else {
      this.hasWebgl = false
    }
  }
}
</script>

<style>
@import 'https://fonts.googleapis.com/css?family=Open+Sans';
@import 'https://fonts.googleapis.com/css?family=Share+Tech+Mono';
@import 'https://fonts.googleapis.com/css?family=Nothing+You+Could+Do';

@import './variables.css';

body {
  background: linear-gradient(0deg, #CCCCCC, #F0F0F0);
  color: var(--color-3);
  min-height: 100vh;
  font-family: var(--font-1);
}

.title {
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;

  & span {
    color: var(--color-1);
    margin-right: 20px;
  }
}

.subtitle {
  color: var(--color-3);
}

.demo {
  padding: 50px 30px;

  & .loading-progress {
    position: absolute;
    top: 0;
    right: 0;
    padding: 30px;
    color: var(--color-1);
    font-size: 18px;
    font-family: var(--font-2);
    padding: 20px 50px;
    margin: 30px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
  }

  & .info-panel {
    width: 100%;
    max-width: 1200px;
    border-radius: 5px;
    background: white;
    color: var(--color-3);
    padding: 20px 20px 5px 20px;
    margin: 20px auto;
    box-shadow: 5px 5px #CCCCCC;

    & .info-panel-text {
      width: 100%;
      margin-bottom: 10px;
      font-size: 12px;

      & a {
        color: var(--color-1);
        transition: color 0.2s ease-in;

        &:hover {
          color: var(--color-1-light);
        }
      }
    }

    & .info-panel-close {
      display: flex;
      align-items: center;
      justify-content: flex-end;

      & .info-panel-close-btn {
        display: flex;
        align-items: center;
        color: var(--color-2);
        transition: color 0.2s ease-in;

        &:hover {
          color: var(--color-1-lighter);
          cursor: pointer;
        }
      }
    }
  }
}

/****************************************************************/
/* MDL overrides */

.mdl-switch__label {
  font-size: 14px !important;
  color: #69707a;
}

.mdl-switch.is-checked {
  & .mdl-switch__thumb {
    background: var(--color-1-light);
  }

  & .mdl-switch__track {
    background: var(--color-1-lighter);
  }
}

.mdl-textfield {
  & .mdl-textfield__input {
    color: var(--color-3);
    border-bottom-color: var(--color-1-light);
    font-family: var(--font-1);
    font-size: 14px;
  }

  & .mdl-textfield__label {
    font-family: var(--font-2);
    color: var(--color-1-light);
  }

  & .mdl-icon-toggle__label {
    color: var(--color-1-light);
  }
}

.mdl-textfield.is-focused {
  & .mdl-textfield__input {
    border-color: var(--color-1);
  }

  & .mdl-textfield__label {
    color: var(--color-1);

    &::after {
      background-color: var(--color-1);
    }
  }

  & .mdl-icon-toggle__label {
    color: var(--color-1);
  }
}

.mdl-menu {
  & .mdl-menu__item {
    font-family: var(--font-2);
    font-size: 14px;
    color: var(--color-2);

    &:hover {
      background-color: var(--color-1-light);
      color: white;
    }
  }
}
</style>
