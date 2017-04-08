import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders

private let MaxFramesInFlight = 3   // use triple buffering

class Visualize {
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue

  var videoTexture: MTLTexture?

  let inflightSemaphore = DispatchSemaphore(value: MaxFramesInFlight)
  var inflightIndex = 0

  var lanczos: MPSImageLanczosScale
  var subtractMeanColor: SubtractMeanColor
  let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
  var img1: MPSImage
  var img2: MPSImage

  var quads: [TexturedQuad] = []

  init(device: MTLDevice, view: MTKView) {
    self.device = device
    commandQueue = device.makeCommandQueue()

    videoTexture = loadTexture(named: "sophie.png")!

    lanczos = MPSImageLanczosScale(device: device)
    subtractMeanColor = SubtractMeanColor(device: device)

    img1 = MPSImage(device: device, imageDescriptor: inputImgDesc)
    img2 = MPSImage(device: device, imageDescriptor: inputImgDesc)

    createQuads(pixelFormat: view.colorPixelFormat)
  }

  func createQuads(pixelFormat: MTLPixelFormat) {
    var quad = TexturedQuad(device: device, pixelFormat: pixelFormat, inflightCount: MaxFramesInFlight)
    quad.position = [112, 112, 0]
    quad.size = 224
    quads.append(quad)

    quad = TexturedQuad(device: device, pixelFormat: pixelFormat, inflightCount: MaxFramesInFlight)
    quad.position = [338, 112, 0]
    quad.size = 224
    quads.append(quad)
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
    }

    if let renderPassDescriptor = view.currentRenderPassDescriptor,
       let currentDrawable = view.currentDrawable {

      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

      quads[0].texture = img1.texture
      quads[0].encode(renderEncoder, matrix: projectionMatrix, for: inflightIndex)

      quads[1].texture = img2.texture
      quads[1].encode(renderEncoder, matrix: projectionMatrix, for: inflightIndex)

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
