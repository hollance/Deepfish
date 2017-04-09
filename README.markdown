# Aunt Emma's ConvNet Diagnostic ToolKit 

This is an iPhone app that applies the VGGNet-16 convolutional neural network to the live camera feed and visualizes what happens after each layer:

TODO pica

Swipe left/right to look at the different layers.

Tap the video preview to switch to a static image of a cat (useful for testing that the neural network code actually works).

> Note: this is alpha software. I quickly threw this together to play with convnet visualizations on my iPhone. There are a lot of things in the code that are not quite kosher yet.

Only tested on the iPhone 6s but the app *should* work on other iPhones too, as long as they have an A8 processor.

## Improvements

Things that need work:

- The further inside the neural network you look, the slower the app becomes. That's because it needs to do more computations. The camera should adjust its FPS to match.

- The camera code does not handle interruptions, going to the background, etc. In a production quality app these sorts of loose ends need to be tied up.

- There may be glitches between how the frames from the video stream are sent to Metal. I did not think this through very carefully yet.

- The UI to swipe between panels needs work (some kind of visual feedback). It's also still glitchy.

- More visualizations, such as deconvolution. Maybe also for the fully-connected layers (at least the probability distribution from the softmax).

## VGGNet

For more information about VGGNet, see [the project page](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and the [paper](http://arxiv.org/pdf/1409.1556):
  
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556

We're using configuration D from the paper, as found in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo):
  
- Input image is 224 x 224 pixels x 3 color channels (RGB).
- All convolution kernels are 3x3.
- All convolution layers use 1-element zero-padding to preserve the width and height of the input volume.
- Convolution is followed by a ReLU.
- All pooling layers are max-pool, size 2, stride 2. These chop the input width and height in half but preserve the depth.
- The fully-connected layers use a ReLU activation, except for the last one which applies the softmax function to produce a probability distribution.

See also my [VGGNet+Metal](https://github.com/hollance/VGGNet-Metal) repo for an example app that uses VGGNet for image classification.

The cat.jpg image is taken from the [Caffe](https://github.com/BVLC/caffe) examples folder.
