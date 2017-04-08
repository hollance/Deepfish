import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders
import CoreMedia
import QuartzCore

let MaxFramesInFlight = 3   // use triple buffering

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var metalView: MTKView!

  var videoCapture: VideoCapture?
  var videoTexture: MTLTexture!

  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var textureLoader: MTKTextureLoader!

  let inflightSemaphore = DispatchSemaphore(value: MaxFramesInFlight)
  var inflightIndex = 0

  var lanczos: MPSImageLanczosScale!
  var subtractMeanColor: SubtractMeanColor!
  let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
  var img1: MPSImage!
  var img2: MPSImage!

  var quads: [TexturedQuad] = []

  override func viewDidLoad() {
    super.viewDidLoad()

    device = MTLCreateSystemDefaultDevice()
    if device != nil {
      textureLoader = MTKTextureLoader(device: device)
      videoTexture = loadTexture(named: "sophie.png")!

      videoCapture = VideoCapture(device: device, delegate: self)

      metalView.clearColor = MTLClearColor(red: 0, green: 0.5, blue: 0.5, alpha: 1)
      metalView.device = device
      metalView.delegate = self

      // Normally the MTKView asks for a redraw 60 times per second. But we're
      // going to give it new textures at a much lower rate, so we'll manually 
      // tell the view when to redraw. Another option would be to redraw at, 
      // say 30 FPS, and only give the view new data when we have it but that's
      // wasteful -- we'll already be burning enough battery as it is.
      metalView.isPaused = true
      metalView.enableSetNeedsDisplay = false

      commandQueue = device.makeCommandQueue()
      lanczos = MPSImageLanczosScale(device: device)
      subtractMeanColor = SubtractMeanColor(device: device)

      img1 = MPSImage(device: device, imageDescriptor: inputImgDesc)
      img2 = MPSImage(device: device, imageDescriptor: inputImgDesc)

      createQuads()
    }
  }

  func createQuads() {
    var quad = TexturedQuad(device: device, view: metalView, inflightCount: MaxFramesInFlight)
    quad.position = [112, 112, 0]
    quad.size = 224
    quads.append(quad)

    quad = TexturedQuad(device: device, view: metalView, inflightCount: MaxFramesInFlight)
    quad.position = [338, 112, 0]
    quad.size = 224
    quads.append(quad)
  }

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  func resizePreviewLayer() {
    videoCapture?.previewLayer?.frame = videoPreview.bounds
  }

  /*
  @IBAction func buttonTapped(_ sender: UIButton) {
    videoCapture.capturePhoto()
  }
  */

  // MARK: - Loading textures

  private func loadTexture(named filename: String) -> MTLTexture? {
    if let url = Bundle.main.url(forResource: filename, withExtension: "") {
      return loadTexture(url: url)
    } else {
      print("Error: could not find image \(filename)")
      return nil
    }
  }

  private func loadTexture(url: URL) -> MTLTexture? {
    do {
      // Note: the SRGB option should be set to false, otherwise the image
      // appears way too dark, since it wasn't actually saved as SRGB.
      return try textureLoader.newTexture(withContentsOf: url, options: [
        MTKTextureLoaderOptionSRGB : NSNumber(value: false)
      ])
    } catch {
      print("Error: could not load texture \(error)")
      return nil
    }
  }

  private func image(from texture: MTLTexture) -> MPSImage {
    // We set featureChannels to 3 because the neural network is only trained
    // on RGB data (the first 3 channels), not alpha (the 4th channel).
    return MPSImage(texture: texture, featureChannels: 3)
  }
}

extension CameraViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didInitializeWithSuccess success: Bool) {
    if success {
      if let previewLayer = capture.previewLayer {
        videoPreview.layer.addSublayer(previewLayer)
        resizePreviewLayer()
      }
      capture.start()
    }
  }

  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
    //print(timestamp)

    // A note on syncing: we set the texture from the video frame on the main
    // thread, and we kick off drawing from the main thread. Drawing may have
    // to wait if there are no free buffers. If in the mean time a new video
    // frame arrives then the closure captures that texture but the closure is
    // not executed until after the next draw has scheduled its stuff. So this
    // should work out fine. The only thing I can think of that might go wrong
    // is if the CVMetalTextureCache loads new data into the same MTLTexture
    // instance that is currently being used for drawing -- but that shouldn't
    // happen.
    DispatchQueue.main.async {
      self.videoTexture = texture
      self.metalView.draw()
    }
  }

  func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
    /*if let texture = texture, let previewImage = previewImage {
      predict(texture: texture, previewImage: previewImage, bgr: true)
    } else {
      imageView.image = nil
    }*/
  }
}

extension CameraViewController: MTKViewDelegate {
  func draw(in view: MTKView) {
    // Block until the next resource is available.
    _ = inflightSemaphore.wait(timeout: .distantFuture)

    let startTime = CACurrentMediaTime()

    // The dimensions of the Metal view in pixels.
    let width = Float(metalView.bounds.width * metalView.contentScaleFactor)
    let height = Float(metalView.bounds.height * metalView.contentScaleFactor)

    // I want the origin (0, 0) to be in the top-left corner, with y positive
    // going down. The default Metal viewport goes from -1 to +1 so we need to
    // scale it by width/2 and height/2, and flip the sign of the height.
    let projectionMatrix = float4x4.scale(to: [2/width, -2/height, 1])
                         * float4x4.translate(to: [-width/2, -height/2, 0])

    let commandBuffer = commandQueue.makeCommandBuffer()

    // NOTE: Normally you'd use MPSTemporaryImage objects for speed and space
    // reasons but since we want to show these textures inside a render pass,
    // we keep to use regular MPSImage objects otherwise they get reused.

    lanczos.encode(commandBuffer: commandBuffer, sourceTexture: videoTexture, destinationTexture: img1.texture)

    subtractMeanColor.encode(commandBuffer: commandBuffer, sourceTexture: img1.texture, destinationTexture: img2.texture)

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

  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    // not implemented
  }
}
