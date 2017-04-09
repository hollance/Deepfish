import Foundation
import Metal
import simd
import UIKit

class Panel {
  var name = ""
  var contentSize = CGSize.zero
  private(set) var quads: [TexturedQuad] = []

  func add(_ quad: TexturedQuad) {
    quads.append(quad)
  }

  subscript(i: Int) -> TexturedQuad {
    return quads[i]
  }

  func set(texture: MTLTexture, forQuadAt index: Int) {
    quads[index].texture = texture
  }

  func set(texture: MTLTexture, max: MTLTexture) {
    // Note: theoretically we could draw all of these quads with a single
    // draw call, since they all use the same texture. But Metal prides 
    // itself on being able to do a lot of draw calls, so this is just as
    // easy. We also don't care about only encoding quads that are visible. 
    for quad in quads {
      quad.texture = texture
      quad.max = max
    }
  }
}
