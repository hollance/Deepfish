import Metal

class SubtractMeanColor {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState

  init(device: MTLDevice) {
    self.device = device
    pipeline = makeFunction(device: device, name: "adjust_mean_bgr")
  }

  func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
    let encoder = commandBuffer.makeComputeCommandEncoder()
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(sourceTexture, at: 0)
    encoder.setTexture(destinationTexture, at: 1)
    encoder.dispatch(pipeline: pipeline, rows: destinationTexture.height, columns: destinationTexture.width)
    encoder.endEncoding()
  }
}
