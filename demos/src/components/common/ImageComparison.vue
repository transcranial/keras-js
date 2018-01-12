<template>
  <div>
    <v-layout row align-center justify-space class="my-3">
      <v-flex><slot name="beforeLabel"></slot></v-flex>
      <v-flex><slot name="afterLabel"></slot></v-flex>
    </v-layout>
    <div
      ref="element"
      class="output-comparison"
      :style="{ height: `${height}px`, width: `${width}px` }"
      @mousemove.prevent="onMouseMove"
      @click="onMouseMove"
    >
      <slot name="after"></slot>
      <div class="before" :style="{ height: `${height}px`, width: `${beforeWidth}px` }">
        <slot name="before"></slot>
      </div>
      <span class="handle" :style="{ left: handleLeftPos }" @mousedown.prevent="onMouseDown"></span>
      <div class="message">
        <slot name="message"></slot>
      </div>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'

export default {
  props: {
    height: { type: Number, default: 0 },
    width: { type: Number, default: 0 }
  },

  data() {
    return {
      position: 0.5,
      isDragging: false
    }
  },

  computed: {
    beforeWidth() {
      return this.width * this.position
    },
    handleLeftPos() {
      return `${this.position * 100}%`
    }
  },

  created() {
    window.addEventListener('mouseup', this.onMouseUp)
  },

  beforeDestroy() {
    window.removeEventListener('mouseup', this.onMouseUp)
  },

  methods: {
    onMouseDown() {
      this.isDragging = true
    },
    onMouseUp(e) {
      e.preventDefault()
      this.isDragging = false
    },
    onMouseMove: _.throttle(function(e) {
      if (this.isDragging) {
        this.position = (e.pageX - this.$refs.element.getBoundingClientRect().left) / this.width
        this.position = Math.max(0, Math.min(1, this.position))
      }
    }, 10)
  }
}
</script>

<style lang="postcss" scoped>
@import '../../variables.css';

.output-comparison {
  display: inline-flex;
  position: relative;
  overflow: hidden;
}

.before {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: 50%;
  overflow: hidden;
}

.handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 4px;
  margin-left: -2px;
  background-color: rgba(255, 255, 255, 0.8);
  cursor: ew-resize;
  transition: background-color 0.3s ease-out;

  &:after {
    position: absolute;
    top: 50%;
    width: 48px;
    height: 48px;
    margin: -24px 0 0 -24px;
    content: '\21d4';
    color: white;
    font-weight: bold;
    font-size: 30px;
    text-align: center;
    line-height: 48px;
    background: #f7ce68;
    background-image: linear-gradient(310deg, #fbab7e 0%, #f7ce68 100%);
    border: 2px solid rgba(255, 255, 255, 0.8);
    border-radius: 50%;
    opacity: 0.8;
    transition: opacity 0.3s ease-out;
  }

  &:hover {
    background-color: white;
    &:after {
      opacity: 1;
    }
  }
}

.message {
  position: absolute;
  top: 5px;
  right: 5px;
}
</style>

