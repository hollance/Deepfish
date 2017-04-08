import Foundation
import Metal
import MetalKit
import simd

struct Vertex {
  var position: float4
  var texCoord: float2
}

class TexturedQuad {
  var pipelineState: MTLRenderPipelineState
  var vertexBuffer: MTLBuffer
  var uniformBuffer: MTLBuffer
  var texture: MTLTexture!

  var position = float3()
  var size: Float = 100

  init(device: MTLDevice, pixelFormat: MTLPixelFormat, inflightCount: Int) {
    let defaultLibrary = device.newDefaultLibrary()!
    let vertexProgram = defaultLibrary.makeFunction(name: "vertexFunc")!
    let fragmentProgram = defaultLibrary.makeFunction(name: "fragmentFunc")!

    let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
    pipelineStateDescriptor.vertexFunction = vertexProgram
    pipelineStateDescriptor.fragmentFunction = fragmentProgram
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = pixelFormat

    try! pipelineState = device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

    let vertexData = [Vertex(position: [-0.5, -0.5, 0, 1], texCoord: [0, 1]),
                      Vertex(position: [ 0.5, -0.5, 0, 1], texCoord: [1, 1]),
                      Vertex(position: [-0.5,  0.5, 0, 1], texCoord: [0, 0]),
                      Vertex(position: [ 0.5,  0.5, 0, 1], texCoord: [1, 0])]

    vertexBuffer = device.makeBuffer(bytes: vertexData, length: MemoryLayout<Vertex>.stride * vertexData.count)
    uniformBuffer = device.makeBuffer(length: MemoryLayout<float4x4>.stride * inflightCount)
  }

  func encode(_ encoder: MTLRenderCommandEncoder, matrix: float4x4, for inflightIndex: Int) {
    // Position the quad. The quad's origin is in its center. Its size goes
    // from -0.5 to +0.5, so we should scale it to the actual size in pixels.
    var matrix = matrix * float4x4.translate(to: position)
                        * float4x4.scale(to: [size, -size, 1])

    // Copy the matrix into the uniform buffer.
    let bufferPointer = uniformBuffer.contents()
    let byteSize = MemoryLayout<float4x4>.stride
    let offset = inflightIndex * byteSize
    memcpy(bufferPointer + offset, &matrix, byteSize)

    encoder.pushDebugGroup("TexturedQuad")
    encoder.setRenderPipelineState(pipelineState)
    encoder.setVertexBuffer(vertexBuffer, offset: 0, at: 0)
    encoder.setVertexBuffer(uniformBuffer, offset: offset, at: 1)
    encoder.setFragmentTexture(texture, at: 0)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.popDebugGroup()
  }
}
