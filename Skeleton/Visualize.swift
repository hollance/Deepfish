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

class Visualize {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue

  var videoTexture: MTLTexture?

  let inflightSemaphore = DispatchSemaphore(value: MaxFramesInFlight)
  var inflightIndex = 0

  let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
  let conv1ImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 64)
  let norm1ImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 64)

  let img1: MPSImage
  let img2: MPSImage
  let img3: MPSImage
  let img4: MPSImage

  let lanczos: MPSImageLanczosScale
  let subtractMeanColor: SubtractMeanColor

  let conv1_1: MPSCNNConvolution  // 224x224x3  input, 64 kernels (3x3x3x64  = 1728  weights + 64 bias)

  let norm1: MPSCNNPoolingMax

  var panels: [Panel] = []
  var activePanelIndex = 0
  var quads: QuadRenderer!

  init(device: MTLDevice, view: MTKView) {
    self.device = device
    commandQueue = device.makeCommandQueue()

    videoTexture = loadTexture(named: "sophie.png")!

    img1 = MPSImage(device: device, imageDescriptor: inputImgDesc)
    img2 = MPSImage(device: device, imageDescriptor: inputImgDesc)
    img3 = MPSImage(device: device, imageDescriptor: conv1ImgDesc)
    img4 = MPSImage(device: device, imageDescriptor: norm1ImgDesc)

    lanczos = MPSImageLanczosScale(device: device)
    subtractMeanColor = SubtractMeanColor(device: device)

    guard let path = Bundle.main.path(forResource: "parameters", ofType: "data"),
          let blob = VGGNetData(path: path) else {
      fatalError("Error loading network parameters")
    }

    conv1_1 = makeConv(device: device, inDepth:   3, outDepth:  64, weights: blob.conv1_1_w, bias: blob.conv1_1_b)

    // The output of a conv layer is between 0 and any positive number (no
    // negative numbers because of the ReLU). We need to scale this down to
    // the range [0, 1]. To find the maximum value in each channel, we can
    // send the conv layer output through a max-pool layer that covers the
    // entire spatial extent. Then in the shader we divide the activations
    // by this max value.
    norm1 = MPSCNNPoolingMax(device: device, kernelWidth: 224, kernelHeight: 224,
                             strideInPixelsX: 224, strideInPixelsY: 224)
    norm1.offset = MPSOffset(x: 112, y: 112, z: 0)

    quads = QuadRenderer(device: device, pixelFormat: view.colorPixelFormat, maxQuads: MaxQuads, inflightCount: MaxFramesInFlight)

    createPanels()
  }

  func createPanels() {
    var panel = Panel()
    panel.name = "Input"
    panel.add(TexturedQuad(position: [112, 112, 0], size: 224))
    panel.add(TexturedQuad(position: [336, 112, 0], size: 224))
    panel.contentSize = CGSize(width: 224*2, height: 224)
    panels.append(panel)

    panel = Panel()
    panel.name = "Conv1.1"
    for j in 0..<16 {
      for i in 0..<4 {
        let y = Float(112 + j * 224)
        let x = Float(112 + i * 224)
        let quad = TexturedQuad(position: [x, y, 0], size: 224)
        quad.isArray = true
        quad.channel = j*4 + i
        panel.add(quad)
      }
    }
    panel.contentSize = CGSize(width: 224*4, height: 224*16)
    panels.append(panel)

    panel = Panel()
    panel.name = "Conv1.2"
    panels.append(panel)

    panel = Panel()
    panel.name = "Pool1"
    panels.append(panel)
  }

  var activePanel: Panel {
    return panels[activePanelIndex]
  }

  func activatePreviousPanel() {
    if activePanelIndex > 0 {
      activePanelIndex -= 1
    }
  }

  func activateNextPanel() {
    if activePanelIndex < panels.count - 1 {
      activePanelIndex += 1
    }
  }

  func draw(in view: MTKView) {
    // Block until the next resource is available.
    _ = inflightSemaphore.wait(timeout: .distantFuture)

    let startTime = CACurrentMediaTime()

    // The dimensions of the Metal view in pixels.
    let width = Float(view.bounds.width * view.contentScaleFactor)
    let height = Float(view.bounds.height * view.contentScaleFactor)

    // I want the origin (0, 0) to be in the top-left corner, with y positive
    // going down. The default Metal viewport goes from -1 to +1 so we need to
    // scale it by width/2 and height/2, and flip the sign of the height.
    let projectionMatrix = float4x4.scale(to: [2/width, -2/height, 1])
                         * float4x4.translate(to: [-width/2, -height/2, 0])

    let commandBuffer = commandQueue.makeCommandBuffer()

    // NOTE: Normally you'd use MPSTemporaryImage objects for speed and space
    // reasons but since we want to show these textures inside a render pass,
    // we keep to use regular MPSImage objects otherwise they get reused.

    if let inputTexture = videoTexture {
      lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: img1.texture)

      subtractMeanColor.encode(commandBuffer: commandBuffer, sourceTexture: img1.texture, destinationTexture: img2.texture)

      if activePanelIndex >= 1 {
        conv1_1.encode(commandBuffer: commandBuffer, sourceImage: img2, destinationImage: img3)
        norm1.encode(commandBuffer: commandBuffer, sourceImage: img3, destinationImage: img4)
      }
    }

    if let renderPassDescriptor = view.currentRenderPassDescriptor,
       let currentDrawable = view.currentDrawable {

      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

      if activePanelIndex == 0 {
        panels[0].set(texture: img1.texture, forQuadAt: 0)
        panels[0].set(texture: img2.texture, forQuadAt: 1)
      }

      if activePanelIndex == 1 {
        panels[1].set(texture: img3.texture, max: img4.texture)
      }

      quads.encode(renderEncoder, quads: activePanel.quads, matrix: projectionMatrix, for: inflightIndex)

      renderEncoder.endEncoding()
      commandBuffer.present(currentDrawable)
    }

    // When this command buffer is done, wake up the next waiting thread.
    commandBuffer.addCompletedHandler { [weak self] commandBuffer in
      let elapsed = CACurrentMediaTime() - startTime
      print("Took \(elapsed) seconds")

      if let strongSelf = self {
        strongSelf.inflightSemaphore.signal()
      }
    }

    inflightIndex = (inflightIndex + 1) % MaxFramesInFlight
    commandBuffer.commit()
  }
}
