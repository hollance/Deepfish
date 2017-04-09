import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders

private let MaxFramesInFlight = 3   // use triple buffering
private let MaxQuads = 100

private func makeConv(device: MTLDevice,
                      inDepth: Int,
                      outDepth: Int,
                      weights: UnsafePointer<Float>,
                      bias: UnsafePointer<Float>) -> MPSCNNConvolution {

  let relu = MPSCNNNeuronReLU(device: device, a: 0)

  let desc = MPSCNNConvolutionDescriptor(kernelWidth: 3,
                                         kernelHeight: 3,
                                         inputFeatureChannels: inDepth,
                                         outputFeatureChannels: outDepth,
                                         neuronFilter: relu)

  let conv = MPSCNNConvolution(device: device,
                               convolutionDescriptor: desc,
                               kernelWeights: weights,
                               biasTerms: bias,
                               flags: .none)
  return conv
}

private func makePool(device: MTLDevice) -> MPSCNNPoolingMax {
  let pool = MPSCNNPoolingMax(device: device,
                              kernelWidth: 2,
                              kernelHeight: 2,
                              strideInPixelsX: 2,
                              strideInPixelsY: 2)
  pool.offset = MPSOffset(x: 1, y: 1, z: 0)
  return pool
}

private func makeNorm(device: MTLDevice, extent: Int) -> MPSCNNPoolingMax {
  // The output of a conv layer is between 0 and any positive number (no
  // negative numbers because of the ReLU). We need to scale this down to
  // the range [0, 1]. To find the maximum value in each channel, we can
  // send the conv layer output through a max-pool layer that covers the
  // entire spatial extent. Then in the shader we divide the activations
  // by this max value. This makes the neural network a little slower but
  // otherwise the visualizations wouldn't make any sense.

  let norm = MPSCNNPoolingMax(device: device,
                              kernelWidth: extent,
                              kernelHeight: extent,
                              strideInPixelsX: extent,
                              strideInPixelsY: extent)
  norm.offset = MPSOffset(x: extent/2, y: extent/2, z: 0)
  return norm
}

class Visualize {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue

  let inflightSemaphore = DispatchSemaphore(value: MaxFramesInFlight)
  var inflightIndex = 0

  // The video frame that we feed into the neural network
  var videoTexture: MTLTexture?
  var channelOrderBGR = false

  let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
  let conv1ImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 64)
  let pool1ImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 64)
  let norm1ImgDesc = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 64)

  // The images that store the outputs of the layers.
  // Normally you'd use MPSTemporaryImage objects for speed and space reasons
  // but since we want to show these textures inside a render pass, we keep to
  // use regular MPSImage objects otherwise these textures get reused. (To be
  // fair, we really only need to keep the final images, so MPSTemporaryImages
  // would actually work. TODO: FIX)
  let imgScaled: MPSImage
  let imgMeanAdjusted: MPSImage
  let imgConv1_1: MPSImage
  let imgNorm1_1: MPSImage
  let imgConv1_2: MPSImage
  let imgNorm1_2: MPSImage
  let imgPool1: MPSImage
  let imgNorm1_3: MPSImage

  // The layers
  let lanczos: MPSImageLanczosScale
  let subtractMeanColor: SubtractMeanColor
  let conv1_1: MPSCNNConvolution  // 224x224x3  input, 64 kernels (3x3x3x64  = 1728  weights + 64 bias)
  let norm1_1: MPSCNNPoolingMax
  let conv1_2: MPSCNNConvolution  // 224x224x64 input, 64 kernels (3x3x64x64 = 36864 weights + 64 bias)
  let norm1_2: MPSCNNPoolingMax
  let pool1  : MPSCNNPoolingMax   // 224x224x64 input -> 112x112x64 output
  let norm1_3: MPSCNNPoolingMax

  var panels: [Panel] = []
  var activePanelIndex = 0
  var renderer: QuadRenderer

  init(device: MTLDevice, view: MTKView) {
    self.device = device
    commandQueue = device.makeCommandQueue()

    imgScaled = MPSImage(device: device, imageDescriptor: inputImgDesc)
    imgMeanAdjusted = MPSImage(device: device, imageDescriptor: inputImgDesc)
    imgConv1_1 = MPSImage(device: device, imageDescriptor: conv1ImgDesc)
    imgNorm1_1 = MPSImage(device: device, imageDescriptor: norm1ImgDesc)
    imgConv1_2 = MPSImage(device: device, imageDescriptor: conv1ImgDesc)
    imgNorm1_2 = MPSImage(device: device, imageDescriptor: norm1ImgDesc)
    imgPool1 = MPSImage(device: device, imageDescriptor: pool1ImgDesc)
    imgNorm1_3 = MPSImage(device: device, imageDescriptor: norm1ImgDesc)

    lanczos = MPSImageLanczosScale(device: device)
    subtractMeanColor = SubtractMeanColor(device: device)

    guard let path = Bundle.main.path(forResource: "parameters", ofType: "data"),
          let blob = VGGNetData(path: path) else {
      fatalError("Error loading network parameters")
    }

    conv1_1 = makeConv(device: device, inDepth:   3, outDepth:  64, weights: blob.conv1_1_w, bias: blob.conv1_1_b)
    norm1_1 = makeNorm(device: device, extent: 224)
    conv1_2 = makeConv(device: device, inDepth:  64, outDepth:  64, weights: blob.conv1_2_w, bias: blob.conv1_2_b)
    norm1_2 = makeNorm(device: device, extent: 224)
    pool1   = makePool(device: device)
    norm1_3 = makeNorm(device: device, extent: 112)

    renderer = QuadRenderer(device: device, pixelFormat: view.colorPixelFormat, maxQuads: MaxQuads, inflightCount: MaxFramesInFlight)

    createPanels()
  }

  func createPanels() {
    var panel = Panel()
    panel.name = "Input"
    panel.extraInfo = "224×224 image, 3 channels"
    panel.add(TexturedQuad(position: [112, 112, 0], size: 224))
    panel.add(TexturedQuad(position: [336, 112, 0], size: 224))
    panel.contentSize = CGSize(width: 224*2, height: 224)
    panels.append(panel)

    panel = Panel()
    panel.name = "Conv1.1"
    panel.extraInfo = "224×224 image, 64 channels"
    panel.configure(extent: 224, rows: 16, columns: 4)
    panels.append(panel)

    panel = Panel()
    panel.name = "Conv1.2"
    panel.extraInfo = "224×224 image, 64 channels"
    panel.configure(extent: 224, rows: 16, columns: 4)
    panels.append(panel)

    panel = Panel()
    panel.name = "Pool1"
    panel.extraInfo = "112×112 image, 64 channels"
    panel.configure(extent: 112, rows: 8, columns: 8)
    panels.append(panel)
  }

  var activePanel: Panel {
    return panels[activePanelIndex]
  }

  func activatePreviousPanel() -> Bool {
    if activePanelIndex > 0 {
      activePanelIndex -= 1
      return true
    }
    return false
  }

  func activateNextPanel() -> Bool {
    if activePanelIndex < panels.count - 1 {
      activePanelIndex += 1
      return true
    }
    return false
  }

  func draw(in view: MTKView, callback: @escaping (CFTimeInterval) -> Void) {
    // Block until the next resource is available.
    _ = inflightSemaphore.wait(timeout: .distantFuture)

    let startTime = CACurrentMediaTime()

    let commandBuffer = commandQueue.makeCommandBuffer()

    // Run the neural network, up to the thing that we're visualizing.
    // This is why the FPS drops the further in the network you look.
    if let inputTexture = videoTexture {
      lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: imgScaled.texture)

      subtractMeanColor.encode(commandBuffer: commandBuffer, sourceTexture: imgScaled.texture, destinationTexture: imgMeanAdjusted.texture, channelOrderBGR: channelOrderBGR)

      if activePanelIndex >= 1 {
        conv1_1.encode(commandBuffer: commandBuffer, sourceImage: imgMeanAdjusted, destinationImage: imgConv1_1)
        if activePanelIndex == 1 {
          norm1_1.encode(commandBuffer: commandBuffer, sourceImage: imgConv1_1, destinationImage: imgNorm1_1)
        }
      }

      if activePanelIndex >= 2 {
        conv1_2.encode(commandBuffer: commandBuffer, sourceImage: imgConv1_1, destinationImage: imgConv1_2)
        if activePanelIndex == 2 {
          norm1_2.encode(commandBuffer: commandBuffer, sourceImage: imgConv1_2, destinationImage: imgNorm1_2)
        }
      }

      if activePanelIndex >= 3 {
        pool1.encode(commandBuffer: commandBuffer, sourceImage: imgConv1_2, destinationImage: imgPool1)
        if activePanelIndex == 3 {
          norm1_3.encode(commandBuffer: commandBuffer, sourceImage: imgPool1, destinationImage: imgNorm1_3)
        }
      }
    }

    if let renderPassDescriptor = view.currentRenderPassDescriptor,
       let currentDrawable = view.currentDrawable {

      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

      if activePanelIndex == 0 {
        panels[activePanelIndex].set(texture: imgScaled.texture, forQuadAt: 0)
        panels[activePanelIndex].set(texture: imgMeanAdjusted.texture, forQuadAt: 1)
      }
      if activePanelIndex == 1 {
        panels[activePanelIndex].set(texture: imgConv1_1.texture, max: imgNorm1_1.texture)
      }
      if activePanelIndex == 2 {
        panels[activePanelIndex].set(texture: imgConv1_2.texture, max: imgNorm1_2.texture)
      }
      if activePanelIndex == 3 {
        panels[activePanelIndex].set(texture: imgPool1.texture, max: imgNorm1_3.texture)
      }

      // The dimensions of the Metal view in pixels.
      let width = Float(view.bounds.width * view.contentScaleFactor)
      let height = Float(view.bounds.height * view.contentScaleFactor)

      // I want the origin (0, 0) to be in the top-left corner, with y positive
      // going down. The default Metal viewport goes from -1 to +1 so we need to
      // scale it by width/2 and height/2, and flip the sign of the height.
      let projectionMatrix = float4x4.scale(to: [2/width, -2/height, 1])
                           * float4x4.translate(to: [-width/2, -height/2, 0])

      renderer.encode(renderEncoder, quads: activePanel.quads, matrix: projectionMatrix, for: inflightIndex)

      renderEncoder.endEncoding()
      commandBuffer.present(currentDrawable)
    }

    // When this command buffer is done, wake up the next waiting thread.
    commandBuffer.addCompletedHandler { [weak self] commandBuffer in
      let elapsed = CACurrentMediaTime() - startTime
      DispatchQueue.main.async { callback(elapsed) }

      if let strongSelf = self {
        strongSelf.inflightSemaphore.signal()
      }
    }

    inflightIndex = (inflightIndex + 1) % MaxFramesInFlight
    commandBuffer.commit()
  }
}
