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

  let img1: MPSImage
  let img2: MPSImage
  let img3: MPSImage

  let lanczos: MPSImageLanczosScale
  let subtractMeanColor: SubtractMeanColor

  let conv1_1: MPSCNNConvolution  // 224x224x3  input, 64 kernels (3x3x3x64  = 1728  weights + 64 bias)

  var quads: QuadRenderer!

  init(device: MTLDevice, view: MTKView) {
    self.device = device
    commandQueue = device.makeCommandQueue()

    videoTexture = loadTexture(named: "sophie.png")!

    img1 = MPSImage(device: device, imageDescriptor: inputImgDesc)
    img2 = MPSImage(device: device, imageDescriptor: inputImgDesc)
    img3 = MPSImage(device: device, imageDescriptor: conv1ImgDesc)

    lanczos = MPSImageLanczosScale(device: device)
    subtractMeanColor = SubtractMeanColor(device: device)

    guard let path = Bundle.main.path(forResource: "parameters", ofType: "data"),
          let blob = VGGNetData(path: path) else {
      fatalError("Error loading network parameters")
    }

    conv1_1 = makeConv(device: device, inDepth:   3, outDepth:  64, weights: blob.conv1_1_w, bias: blob.conv1_1_b)

    quads = QuadRenderer(device: device, pixelFormat: view.colorPixelFormat, maxQuads: MaxQuads, inflightCount: MaxFramesInFlight)
    createQuads()
  }

  func createQuads() {
    quads.add(TexturedQuad(position: [112, 112, 0], size: 224))
    quads.add(TexturedQuad(position: [338, 112, 0], size: 224))

    for i in 0..<3 {
      for j in 0..<3 {
        let y = Float(112 + (i + 1) * (224 + 2))
        let x = Float(112 + j * (224 + 2))
        let quad = TexturedQuad(position: [x, y, 0], size: 224)
        quad.isArray = true
        quad.channel = i*3 + j
        quads.add(quad)
      }
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

      conv1_1.encode(commandBuffer: commandBuffer, sourceImage: img2, destinationImage: img3)
    }

    if let renderPassDescriptor = view.currentRenderPassDescriptor,
       let currentDrawable = view.currentDrawable {

      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

      quads[0].texture = img1.texture
      quads[1].texture = img2.texture
      for i in 0..<9 {
        quads[i + 2].texture = img3.texture
      }
      quads.encode(renderEncoder, matrix: projectionMatrix, for: inflightIndex)

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
