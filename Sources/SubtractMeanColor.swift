import Metal

class SubtractMeanColor {
  let device: MTLDevice
  let pipelineRGB: MTLComputePipelineState
  let pipelineBGR: MTLComputePipelineState

  init(device: MTLDevice) {
    self.device = device
    pipelineRGB = makeFunction(device: device, name: "adjust_mean_rgb")
    pipelineBGR = makeFunction(device: device, name: "adjust_mean_bgr")
  }

  func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture, channelOrderBGR: Bool) {
    let encoder = commandBuffer.makeComputeCommandEncoder()
    encoder.setComputePipelineState(channelOrderBGR ? pipelineBGR: pipelineRGB)
    encoder.setTexture(sourceTexture, at: 0)
    encoder.setTexture(destinationTexture, at: 1)
    encoder.dispatch(pipeline: channelOrderBGR ? pipelineBGR: pipelineRGB, rows: destinationTexture.height, columns: destinationTexture.width)
    encoder.endEncoding()
  }
}
