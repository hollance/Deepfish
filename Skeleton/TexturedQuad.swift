import Foundation
import Metal
import MetalKit
import simd

struct Vertex {
  var position: float4
  var texCoord: float2
}

class TexturedQuad {
  var position = float3()
  var size: Float = 100
  var texture: MTLTexture?

  init(position: float3, size: Float) {
    self.position = position
    self.size = size
  }
}

class QuadRenderer {
  let maxQuads: Int
  let inflightCount: Int
  let pipelineState: MTLRenderPipelineState
  let vertexBuffer: MTLBuffer
  let uniformBuffer: MTLBuffer

  private(set) var quads: [TexturedQuad] = []

  init(device: MTLDevice, pixelFormat: MTLPixelFormat, maxQuads: Int, inflightCount: Int) {
    self.maxQuads = maxQuads
    self.inflightCount = inflightCount

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
    uniformBuffer = device.makeBuffer(length: MemoryLayout<float4x4>.stride * maxQuads * inflightCount)
  }

  func add(_ quad: TexturedQuad) {
    if quads.count < maxQuads {
      quads.append(quad)
    } else {
      print("Too many quads!")
    }
  }

  subscript(i: Int) -> TexturedQuad {
    return quads[i]
  }

  func encode(_ encoder: MTLRenderCommandEncoder, matrix: float4x4, for inflightIndex: Int) {
    for (quadIndex, quad) in quads.enumerated() {

      // Position the quad. The quad's origin is in its center. Its size goes
      // from -0.5 to +0.5, so we should scale it to the actual size in pixels.
      var matrix = matrix * float4x4.translate(to: quad.position)
                          * float4x4.scale(to: [quad.size, -quad.size, 1])

      // Copy the matrix into the uniform buffer.
      let bufferPointer = uniformBuffer.contents()
      let byteSize = MemoryLayout<float4x4>.stride
      let offset = (inflightIndex*inflightCount + quadIndex)*byteSize
      memcpy(bufferPointer + offset, &matrix, byteSize)

      encoder.pushDebugGroup("TexturedQuad")
      encoder.setRenderPipelineState(pipelineState)
      encoder.setVertexBuffer(vertexBuffer, offset: 0, at: 0)
      encoder.setVertexBuffer(uniformBuffer, offset: offset, at: 1)
      encoder.setFragmentTexture(quad.texture, at: 0)
      encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
      encoder.popDebugGroup()
    }
  }
}
