import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../components/Home'
import MnistCnn from '../components/models/MnistCnn'
import MnistVae from '../components/models/MnistVae'
import MnistAcgan from '../components/models/MnistAcgan'
import Resnet50 from '../components/models/Resnet50'
import InceptionV3 from '../components/models/InceptionV3'
import DenseNet121 from '../components/models/DenseNet121'
import SqueezenetV1 from '../components/models/SqueezenetV1'
import ImdbBidirectionalLstm from '../components/models/ImdbBidirectionalLstm'
import ImageSuperResolution from '../components/models/ImageSuperResolution'

Vue.use(VueRouter)

const router = new VueRouter({
  routes: [
    { path: '/', component: Home },
    { path: '/mnist-cnn', component: MnistCnn },
    { path: '/mnist-vae', component: MnistVae },
    { path: '/mnist-acgan', component: MnistAcgan },
    { path: '/resnet50', component: Resnet50 },
    { path: '/inception-v3', component: InceptionV3 },
    { path: '/densenet121', component: DenseNet121 },
    { path: '/squeezenet-v1.1', component: SqueezenetV1 },
    { path: '/imdb-bidirectional-lstm', component: ImdbBidirectionalLstm },
    { path: '/image-super-resolution', component: ImageSuperResolution }
  ]
})

export default router
