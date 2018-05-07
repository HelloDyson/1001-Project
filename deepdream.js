//last update 180503

// import * as tf from '@tensorflow/tfjs';
const tf = require('@tensorflow/tfjs'); //sometimes require not working so well???
import { IMAGENET_CLASSES } from './imagenet_classes';

//----------------------------------------------------------------------------------------------------------
// part1 - loading model, warmup model, prepare data
// code based on : https://github.com/google/deepdream/blob/master/dream.ipynb (hard to learn)
// inception_v3 model is converted using tfjs
// KERAS MODEL arachitecture: https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
// weights from here: https://github.com/fchollet/deep-learning-models/releases/tag/v0.2
//
//
//
// For this project: 
// At the first time, i was using Synaptic.js, it can build DNN models in browser but it can only run on small data
// and it doesnot have good API to load pretrained model.
// Then I tryied caffeJS (which is based on CONVELUJS), it continoiously producing errors, don't know why.
// At last, i changed to Tensorflow.js. Much better.
// This js file is not working that well.
// Need more research......
// dyson
//
//
//----------------------------------------------------------------------------------------------------------
const Inception_v3_path =
  // load the models
  //this is a model tranfered from google Inception_v3
  './deepdeep_model/model.json';

const IMAGE_SIZE = 300;

//innit the model and warm it up
let inception;
const loadModel = async () => {
  status('Loading model...');
  inception = await tf.loadModel(Inception_v3_path);
  // Warmup the model. 
  inception.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  //clear status
  status('');
};

//prepare the img data to fit the data input type of inception model
//and return a ready img data
function prepareInput(imgElement) {
  // tf.fromPixels() returns a Tensor from an image element.
  const img = tf.fromPixels(imgElement).toFloat();
  const offset = tf.scalar(127.5);
  // Normalize the image from [0, 255] to [-1, 1].
  const normalized = img.sub(offset).div(offset);
  // Reshape to a single-element batch so we can pass it to predict.
  const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  return batched;
}

// loadModel().then(prepareInput);

//----------------------------------------------------------------------------------------------------------
// part2 - Producing dreams
// Making the "dream" images is very simple. 
// Essentially it is just a gradient ascent process that tries to maximize
// the L2 norm of activations of a particular DNN layer. 
// Here are a few simple tricks that we found useful for getting good images:
// ------- offset image by a random jitter
// ------- normalize the magnitude of gradient ascent steps
// ------- apply ascent across multiple scales (octaves)
//----------------------------------------------------------------------------------------------------------

// the mean value can be found in train_val.prototxt
var mean = [104.0, 116.0, 122.0];

// @src https://github.com/google/deepdream/blob/master/dream.ipynb
var objective_L2 = function (dst) {
  dst.out_act.dw.set(dst.out_act.w);
}

function make_step(net, in_data, params) {
  params = params || {};
  var step_size = params.step_size || 1.5;
  var jitter = params.jitter || 32;
  var clip_data = params.clip_data === undefined ? true : clip_data;
  var end = params.end || 'inception_4c/output';
  var objective = params.objective || objective_L2;

  var src = inception.getLayer('data');
  var dst = inception.getLayer(end);

  var ox = Numjs.randi(-jitter, jitter + 1);
  var oy = Numjs.randi(-jitter, jitter + 1);

  // apply jitter shift
  in_data = in_data.roll(ox, oy);

  inception.forward(in_data, { end: end })
  objective(dst)  // specify the optimization objective
  inception.backward(undefined, { start: end })

  var out_data = src.out_act;
  var diff = out_data.dw;

  var mean_diff = 0.0;
  for (var i = 0, len = diff.length; i < len; i++) {
    mean_diff += Math.abs(diff[i]);
  }
  mean_diff /= diff.length;

  // apply normalized ascent step to the input image
  out_data.w = Numjs.addScaled(out_data.w, diff, step_size / mean_diff);

  // unshift image
  out_data = out_data.roll(-ox, -oy);

  if (clip_data) {
    bias = mean;
    for (var d = 0; d < out_data.depth; d++) {
      for (var x = 0; x < out_data.sx; x++) {
        for (var y = 0; y < out_data.sy; y++) {
          var dval = nj.clip(out_data.get(x, y, d), -bias[d], 255 - bias[d]);
          out_data.set(x, y, d, dval);
        }
      }
    }
  }

  return out_data;
}


//----------------------------------------------------------------------------------------------------------
// part3 - gradient ascent through different scales. the scales "octaves".
//----------------------------------------------------------------------------------------------------------

function deepdream(inception, vol, params) {
  params = params || {};
  var iter_n = params.iter_n || 10;
  var octave_n = params.octave_n || 4;
  var octave_scale = params.octave_scale || 1.4;
  var end = params.end || 'inception_4c/output';

  // prepare base images for all octaves
  var octaves = [vol];
  for (var i = 0; i < octave_n; i++) {
    octaves.push(octaves[octaves.length - 1].zoom(1.0 / octave_scale, 1.0 / octave_scale, 1));
  }

  // allocate image for network-produced details
  var detail = octaves[octaves.length - 1].cloneAndZero();

  for (var i = octave_n; i >= 0; i--) {
    var octave = octave_n - i;
    var octave_base = octaves[i];
    var w = octave_base.sx;
    var h = octave_base.sy;

    if (octave > 0) {
      // upscale details from the previous octave
      var w1 = detail.sx;
      var h1 = detail.sy;
      detail = detail.zoom(w / w1, h / h1, 1);
    }

    // extract details produced on the current octave
    vol = octave_base.clone();
    vol.w = nj.add(vol.w, detail.w);

    // For now the model dimensions are fixed
    // so lets update them
    model.setInputDimensions(vol.sx, vol.sy);

    for (var j = 0; j < iter_n; j++) {
      vol = make_step(inception, vol, { end: end });
    }
    // extract details produced on the current octave
    detail = vol.clone()
    detail.w = nj.sub(detail.w, octave_base.w);
  }

  // returning the resulting image
  return vol;
}

self.onmessage = function (e) {
  var vol = inception.Vol.fromJSON(e.data.input);
  var params = e.data.params;

  // Perform a deepdream
  vol = deepdream(model, vol, params);

  // in python
  // # save imgs
  // output_file = 'output' + str(num) + '.jpg'
  // cv2.imwrite(output_file, img)  ---- using canvas?????
  // print(output_file)
  // # select the layer    
  // layer = 'mixed4d'

  // # load input
  // for i in range(1):
  //     input_img = np.float32(cv2.imread("frames/frame-001_"+str(i+40)+".jpg"))
  //     deep_dream(tf.square(graph.get_tensor_by_name("import/%s:0"%layer)),input_img,num=i+40)
}