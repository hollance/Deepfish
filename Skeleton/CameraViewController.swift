import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders
import CoreMedia

let MaxFramesInFlight = 3   // use triple buffering

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var metalView: MTKView!

  var videoCapture: VideoCapture?

  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var textureLoader: MTKTextureLoader!
  var quad: TexturedQuad!

  var lanczos: MPSImageLanczosScale!
  var videoTexture: MTLTexture!
  var matrix = float4x4()

  let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)

  let inflightSemaphore = DispatchSemaphore(value: MaxFramesInFlight)
  var inflightIndex = 0

  override func viewDidLoad() {
    super.viewDidLoad()

    device = MTLCreateSystemDefaultDevice()
    if device != nil {
      textureLoader = MTKTextureLoader(device: device)

      videoCapture = VideoCapture(device: device, delegate: self)

      metalView.clearColor = MTLClearColor(red: 0, green: 0.5, blue: 0.5, alpha: 1)
      metalView.device = device
      metalView.delegate = self
      metalView.isPaused = true
      metalView.enableSetNeedsDisplay = false

      commandQueue = device.makeCommandQueue()

      quad = TexturedQuad(device: device, view: metalView, inflightCount: MaxFramesInFlight)
      videoTexture = loadTexture(named: "sophie.png")!
      lanczos = MPSImageLanczosScale(device: device)
    }
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

  // MARK: - Animations

  var r: Float = 0
  func update() {
    //r += 0.05

    // The dimensions of the Metal view in pixels.
    let width = Float(metalView.bounds.width * metalView.contentScaleFactor)
    let height = Float(metalView.bounds.height * metalView.contentScaleFactor)

    // I want the origin (0, 0) to be in the top-left corner, with y positive
    // going down. The default Metal viewport goes from -1 to +1 so we need to
    // scale it by width/2 and height/2, and flip the sign of the height.
    let projectionMatrix = float4x4.scale(to: [2/width, -2/height, 1])
                         * float4x4.translate(to: [-width/2, -height/2, 0])

    // Position the quad. The quad's origin is in its center. Its size goes
    // from -0.5 to +0.5, so we should scale it to the actual size in pixels.
    matrix = projectionMatrix
           * float4x4.translate(to: [112, 112, 0])
           * float4x4.scale(to: [224, -224, 1])
           * float4x4.rotate(to: [0, 0, r])
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

    update()

    let commandBuffer = commandQueue.makeCommandBuffer()

    if let renderPassDescriptor = view.currentRenderPassDescriptor,
       let currentDrawable = view.currentDrawable {

      let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: inputImgDesc)
      lanczos.encode(commandBuffer: commandBuffer, sourceTexture: videoTexture, destinationTexture: img1.texture)

      let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

      quad.texture = img1.texture
      quad.encode(renderEncoder, matrix: matrix, for: inflightIndex)
      img1.readCount -= 1

      renderEncoder.endEncoding()
      commandBuffer.present(currentDrawable)
    }

    // When this command buffer is done, wake up the next waiting thread.
    commandBuffer.addCompletedHandler { [weak self] commandBuffer in
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
