import Foundation
import Metal
import MetalKit
import simd

struct Vertex {
  var position: float4
  var texCoord: float2
}

struct VertexUniform {
  var matrix = float4x4.identity
}

struct FragmentUniform {
  var textureIndex: Int16 = 0
  var channelIndex: Int16 = 0
}

class TexturedQuad {
  var position = float3()
  var size: Float = 100     // in pixels, not points!
  var texture: MTLTexture?
  var max: MTLTexture?      // for normalizing
  var isArray = false
  var channel = 0           // -1 means display as RGBA

  init(position: float3, size: Float) {
    self.position = position
    self.size = size
  }
}

class QuadRenderer {
  let maxQuads: Int
  let inflightCount: Int
  let pipeline: MTLRenderPipelineState
  let pipelineA: MTLRenderPipelineState
  let vertexBuffer: MTLBuffer
  let vertexUniformBuffer: MTLBuffer
  let fragmentUniformBuffer: MTLBuffer

  init(device: MTLDevice, pixelFormat: MTLPixelFormat, maxQuads: Int, inflightCount: Int) {
    self.maxQuads = maxQuads
    self.inflightCount = inflightCount

    let library = device.newDefaultLibrary()!
    let vertexProgram = library.makeFunction(name: "vertexFunc")!
    let fragmentProgram = library.makeFunction(name: "fragmentFunc")!
    let fragmentProgramA = library.makeFunction(name: "fragmentFuncA")!

    let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
    pipelineStateDescriptor.vertexFunction = vertexProgram
    pipelineStateDescriptor.fragmentFunction = fragmentProgram
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = pixelFormat

    try! pipeline = device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

    pipelineStateDescriptor.fragmentFunction = fragmentProgramA
    try! pipelineA = device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

    let vertexData = [Vertex(position: [-0.5, -0.5, 0, 1], texCoord: [0, 1]),
                      Vertex(position: [ 0.5, -0.5, 0, 1], texCoord: [1, 1]),
                      Vertex(position: [-0.5,  0.5, 0, 1], texCoord: [0, 0]),
                      Vertex(position: [ 0.5,  0.5, 0, 1], texCoord: [1, 0])]

    vertexBuffer = device.makeBuffer(bytes: vertexData, length: MemoryLayout<Vertex>.stride * vertexData.count)
    vertexUniformBuffer = device.makeBuffer(length: MemoryLayout<VertexUniform>.stride * maxQuads * inflightCount)
    fragmentUniformBuffer = device.makeBuffer(length: MemoryLayout<FragmentUniform>.stride * maxQuads * inflightCount)
  }

  func encode(_ encoder: MTLRenderCommandEncoder, quads: [TexturedQuad], matrix: float4x4, for inflightIndex: Int) {
    for (quadIndex, quad) in quads.enumerated() {
      if quadIndex >= maxQuads {
        print("Too many quads!")
        return
      }

      // Position the quad. The quad's origin is in its center. Its size goes
      // from -0.5 to +0.5, so we should scale it to the actual size in pixels.
      var vertexUniform = VertexUniform()
      vertexUniform.matrix = matrix * float4x4.translate(to: quad.position)
                                    * float4x4.scale(to: [quad.size, -quad.size, 1])

      // Copy the matrix into the uniform buffer.
      let vertexUniformSize = MemoryLayout<VertexUniform>.stride
      let vertexOffset = (inflightIndex*inflightCount + quadIndex)*vertexUniformSize
      memcpy(vertexUniformBuffer.contents() + vertexOffset, &vertexUniform, vertexUniformSize)

      var fragmentUniform = FragmentUniform()
      fragmentUniform.textureIndex = Int16(quad.channel) / 4
      fragmentUniform.channelIndex = Int16(quad.channel) - fragmentUniform.textureIndex*4

      let fragmentUniformSize = MemoryLayout<FragmentUniform>.stride
      let fragmentOffset = (inflightIndex*inflightCount + quadIndex)*fragmentUniformSize
      memcpy(fragmentUniformBuffer.contents() + fragmentOffset, &fragmentUniform, fragmentUniformSize)

      encoder.pushDebugGroup("TexturedQuad")
      encoder.setRenderPipelineState(quad.isArray ? pipelineA : pipeline)
      encoder.setVertexBuffer(vertexBuffer, offset: 0, at: 0)
      encoder.setVertexBuffer(vertexUniformBuffer, offset: vertexOffset, at: 1)
      encoder.setFragmentBuffer(fragmentUniformBuffer, offset: fragmentOffset, at: 0)
      encoder.setFragmentTexture(quad.texture, at: 0)
      encoder.setFragmentTexture(quad.max, at: 1)
      encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
      encoder.popDebugGroup()
    }
  }
}
