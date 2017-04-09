import Foundation
import simd

extension float4x4 {
  static let identity = float4x4(diagonal: float4(1))

  static func translate(to position: float3) -> float4x4 {
    var matrix = self.identity
    matrix[3, 0] = position.x
    matrix[3, 1] = position.y
    matrix[3, 2] = position.z
    return matrix
  }

  static func scale(to scale: Float) -> float4x4 {
    var matrix = self.identity
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    matrix[3, 3] = 1
    return matrix
  }

  static func scale(to scale: float3) -> float4x4 {
    var matrix = self.identity
    matrix[0, 0] = scale.x
    matrix[1, 1] = scale.y
    matrix[2, 2] = scale.z
    matrix[3, 3] = 1
    return matrix
  }

  static func rotate(to rot: float3) -> float4x4 {
    var matrix = self.identity
    matrix[0, 0] =  cos(rot.y) * cos(rot.z)
    matrix[1, 0] =  cos(rot.z) * sin(rot.x) * sin(rot.y) - cos(rot.x) * sin(rot.z)
    matrix[2, 0] =  cos(rot.x) * cos(rot.z) * sin(rot.y) + sin(rot.x) * sin(rot.z)
    matrix[0, 1] =  cos(rot.y) * sin(rot.z)
    matrix[1, 1] =  cos(rot.x) * cos(rot.z) + sin(rot.x) * sin(rot.y) * sin(rot.z)
    matrix[2, 1] = -cos(rot.z) * sin(rot.x) + cos(rot.x) * sin(rot.y) * sin(rot.z)
    matrix[0, 2] = -sin(rot.y)
    matrix[1, 2] =  cos(rot.y) * sin(rot.x)
    matrix[2, 2] =  cos(rot.x) * cos(rot.y)
    matrix[3, 3] =  1
    return matrix
  }
}
