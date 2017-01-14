import weblas from 'weblas/dist/weblas';
import Model from './Model';
import Tensor from './Tensor';
import * as activations from './activations';
import * as layers from './layers';
import * as testUtils from './utils/testUtils';

window.weblas = weblas;

export { Model, Tensor, activations, layers, testUtils };
