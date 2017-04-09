import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders
import CoreMedia
import QuartzCore

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var scrollView: UIScrollView!
  @IBOutlet weak var metalView: MTKView!

  var videoCapture: VideoCapture?
  var visualize: Visualize?

  var device: MTLDevice!

  override func viewDidLoad() {
    super.viewDidLoad()

    device = MTLCreateSystemDefaultDevice()
    if device != nil {
      videoCapture = VideoCapture(device: device, delegate: self)
      visualize = Visualize(device: device, view: metalView)

      metalView.clearColor = MTLClearColor(red: 20/255, green: 30/255, blue: 40/255, alpha: 1)
      metalView.device = device
      metalView.delegate = self

      // Normally the MTKView asks for a redraw 60 times per second. But we're
      // going to give it new textures at a much lower rate, so we'll manually 
      // tell the view when to redraw. Another option would be to redraw at, 
      // say 30 FPS, and only give the view new data when we have it but that's
      // wasteful -- we'll already be burning enough battery as it is.
      metalView.isPaused = true
      metalView.enableSetNeedsDisplay = false
    }
  }

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()

    let width: CGFloat = 224*4 / view.window!.screen.scale
    var frame = scrollView.frame
    frame.origin.x = view.bounds.width - width
    frame.size.width = width
    frame.size.height = view.bounds.height
    scrollView.frame = frame

    metalView.frame = CGRect(x: 0, y: 0, width: width, height: 224*17 / view.window!.screen.scale)
    scrollView.contentSize = metalView.bounds.size

    frame = videoPreview.frame
    frame.size.width = scrollView.frame.origin.x
    frame.size.height = frame.size.width * 3 / 4
    videoPreview.frame = frame

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
      self.visualize?.videoTexture = texture
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
    visualize?.draw(in: view)
  }

  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    // not implemented
  }
}
