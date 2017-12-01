<template>
  <aside class="menu">
    <div class="logo"><a href="#/"><img src="assets/logo.svg" /></a></div>
    <p class="menu-label">Demos</p>
    <ul class="menu-list">
      <li :class="{ active: currentView === 'mnist-cnn' }">
        <a href="#/mnist-cnn">
          <span class="menu-item-heading">Basic Convnet</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'mnist-vae' }">
        <a href="#/mnist-vae">
          <span class="menu-item-heading">Convolutional VAE</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'mnist-acgan' }">
        <a href="#/mnist-acgan">
          <span class="menu-item-heading">AC-GAN</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'resnet50' }">
        <a href="#/resnet50">
          <span class="menu-item-heading">ResNet-50</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'inception-v3' }">
        <a href="#/inception-v3">
          <span class="menu-item-heading">Inception v3</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'densenet121' }">
        <a href="#/densenet121">
          <span class="menu-item-heading">DenseNet-121</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'squeezenet-v1.1' }">
        <a href="#/squeezenet-v1.1">
          <span class="menu-item-heading">SqueezeNet v1.1</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'imdb-bidirectional-lstm' }">
        <a href="#/imdb-bidirectional-lstm">
          <span class="menu-item-heading">Bidirectional LSTM</span>
          <span class="menu-item-subheading">IMDB</span>
        </a>
      </li>
    </ul>
    <p class="menu-label">Links</p>
    <ul class="menu-list github">
      <li>
        <a href="https://github.com/transcranial/keras-js" target="_blank" rel="noopener noreferrer">
          <span class="menu-item-heading">
            <svg height="24" width="24" class="github-logo" viewBox="0 0 16 16" version="1.1" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
            GitHub repo
          </span>
        </a>
      </li>
    </ul>
    <p class="menu-label">Contact</p>
    <ul class="menu-list contact">
      <li>
        <a href="https://github.com/transcranial" target="_blank" rel="noopener noreferrer">Leon Chen <@transcranial></a>
      </li>
      <li></li>
    </ul>
  </aside>
</template>

<script>
export default {
  props: {
    currentView: { type: String, default: 'home' }
  }
}
</script>

<style scoped lang="postcss">
@import '../variables.css';

.menu {
  font-family: var(--font-sans-serif);
  padding: 20px 40px;
  background: whitesmoke;
}

.logo {
  & img {
    max-width: 100%;
    max-height: 100%;
  }
}

.menu-label {
  user-select: none;
  cursor: default;
  font-size: 11px;
  color: var(--color-lightgray);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin: 11px 0;
}

.menu-list {
  list-style: none;

  & li {
    color: var(--color-lightgray);
    border-left: 2px solid whitesmoke;
    margin-bottom: 5px;
  }

  & li.active {
    border-left: 2px solid var(--color-green);
  }

  & a {
    padding: 7px 11px;
    color: var(--color-darkgray);
    text-decoration: none;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;

    &:hover {
      color: var(--color-green);
      background-color: whitesmoke;

      & .github-logo {
        fill: var(--color-green);
      }
    }

    & span.menu-item-heading {
      display: flex;
      align-items: center;
      margin-right: 10px;
      font-size: 14px;
    }

    & span.menu-item-subheading {
      color: #999999;
      font-size: 10px;
    }
  }
}

.menu-list.github,
.menu-list.contact {
  & li {
    padding: 5px 10px;
    font-size: 14px;
  }

  & a {
    color: var(--color-green);
    padding: 0;
    display: inline-flex;
    background-color: none;
    transition: color 0.2s ease-in-out;

    & .github-logo {
      fill: var(--color-green);
      transition: fill 0.2s ease-in-out;
      margin-right: 5px;
    }

    &:hover {
      color: var(--color-green-light);
      background: none;

      & .github-logo {
        fill: var(--color-green-light);
      }
    }
  }
}
</style>
