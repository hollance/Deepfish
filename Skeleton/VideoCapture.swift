import UIKit
import AVFoundation
import CoreVideo
import Metal

/*
  Simple interface to the iPhone's camera.
  
  TODO: 
    - need to handle going to background and other interruptions
    - need more flexibility in choosing format, FPS, etc
*/

public protocol VideoCaptureDelegate: class {
  func videoCapture(_ capture: VideoCapture, didInitializeWithSuccess success: Bool)
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime)
  func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?)
}

public class VideoCapture: NSObject {

  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?

  let device: MTLDevice
  var textureCache: CVMetalTextureCache?
  let captureSession = AVCaptureSession()
  let videoOutput = AVCaptureVideoDataOutput()
	let photoOutput = AVCapturePhotoOutput()
  let queue = DispatchQueue(label: "net.machinethink.camera-queue")

  var lastTimestamp = CMTime()
  let FPS = 15

  public init(device: MTLDevice, delegate: VideoCaptureDelegate) {
    self.device = device
    self.delegate = delegate
    super.init()

    queue.async {
      let success = self.setUpCamera()
      DispatchQueue.main.async {
        self.delegate?.videoCapture(self, didInitializeWithSuccess: success)
      }
    }
  }

  func setUpCamera() -> Bool {
    guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
      print("Error: could not create a texture cache")
      return false
    }

    captureSession.beginConfiguration()
    captureSession.sessionPreset = AVCaptureSessionPresetMedium

    guard let captureDevice = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) else {
      print("Error: no video devices available")
      return false
    }

    guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
      print("Error: could not create AVCaptureDeviceInput")
      return false
    }

    if captureSession.canAddInput(videoInput) {
      captureSession.addInput(videoInput)
    }

    if let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession) {
      previewLayer.videoGravity = AVLayerVideoGravityResizeAspect
      previewLayer.connection?.videoOrientation = .landscapeRight
      self.previewLayer = previewLayer
    }

    let settings: [String : Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: queue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }

    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
    }

    captureSession.commitConfiguration()

    /*
    // This sets the framerate to the minimum available. However, this
    // also changes the framerate of the preview, which is ugly.
    if let _ = try? captureDevice.lockForConfiguration() {
      let ranges = captureDevice.activeFormat.videoSupportedFrameRateRanges as! [AVFrameRateRange]
      let minFrameRate = Int32(ranges.first!.minFrameRate)

      captureDevice.activeVideoMaxFrameDuration = CMTimeMake(1, minFrameRate)
      captureDevice.activeVideoMinFrameDuration = CMTimeMake(1, minFrameRate)
      captureDevice.unlockForConfiguration()
    }
    */

    return true
  }

  public func start() {
    if !captureSession.isRunning {
      captureSession.startRunning()
    }
  }

  public func stop() {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }
  }

  /* Captures a single frame of the camera input. */
  public func capturePhoto() {
    let settings = AVCapturePhotoSettings(format: [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ])

    settings.previewPhotoFormat = [
      kCVPixelBufferPixelFormatTypeKey as String: settings.availablePreviewPhotoPixelFormatTypes[0],
      kCVPixelBufferWidthKey as String: 480,
      kCVPixelBufferHeightKey as String: 360,
    ]

    photoOutput.capturePhoto(with: settings, delegate: self)
  }

  func convertToMTLTexture(sampleBuffer: CMSampleBuffer?) -> MTLTexture? {
    if let textureCache = textureCache,
       let sampleBuffer = sampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)

      var texture: CVMetalTexture?
      CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
          imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)

      if let texture = texture {
        return CVMetalTextureGetTexture(texture)
      }
    }
    return nil
  }

  func convertToUIImage(sampleBuffer: CMSampleBuffer?) -> UIImage? {
    if let sampleBuffer = sampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)
      let rect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))

      let ciImage = CIImage(cvPixelBuffer: imageBuffer)
      let ciContext = CIContext(options: nil)
      if let cgImage = ciContext.createCGImage(ciImage, from: rect) {
        return UIImage(cgImage: cgImage)
      }
    }
    return nil
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  public func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {

    // Because lowering the capture device's FPS looks ugly in the preview, we
    // capture at full speed but only call the delegate at its desired framerate.

    let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    let deltaTime = timestamp - lastTimestamp
    if true || deltaTime >= CMTimeMake(1, Int32(FPS)) {
      lastTimestamp = timestamp

      let texture = convertToMTLTexture(sampleBuffer: sampleBuffer)
      delegate?.videoCapture(self, didCaptureVideoTexture: texture, timestamp: timestamp)
    }
  }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
  public func capture(_ captureOutput: AVCapturePhotoOutput,
                      didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?,
                      previewPhotoSampleBuffer: CMSampleBuffer?,
                      resolvedSettings: AVCaptureResolvedPhotoSettings,
                      bracketSettings: AVCaptureBracketedStillImageSettings?,
                      error: Error?) {

    var imageTexture: MTLTexture?
    var previewImage: UIImage?

    if error == nil {
      imageTexture = convertToMTLTexture(sampleBuffer: photoSampleBuffer)
      previewImage = convertToUIImage(sampleBuffer: previewPhotoSampleBuffer)
    }
    delegate?.videoCapture(self, didCapturePhotoTexture: imageTexture, previewImage: previewImage)
  }
}
