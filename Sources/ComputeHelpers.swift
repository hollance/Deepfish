import Foundation
import Metal
import MetalPerformanceShaders

/**
  Helper function that creates a pipeline for a compute kernel.
*/
func makeFunction(library: MTLLibrary, name: String) -> MTLComputePipelineState {
  do {
    guard let kernelFunction = library.makeFunction(name: name) else {
      fatalError("Could not load compute function '\(name)'")
    }
    return try library.device.makeComputePipelineState(function: kernelFunction)
  } catch {
    fatalError("Could not create compute pipeline for function '\(name)'")
  }
}

func makeFunction(device: MTLDevice, name: String) -> MTLComputePipelineState {
  guard let library = device.newDefaultLibrary() else {
    fatalError("Could not load Metal library")
  }
  return makeFunction(library: library, name: name)
}

extension MTLComputeCommandEncoder {
  /**
    Dispatches a compute kernel on a 2-dimensional grid.
    
    - Parameters:
      - rows: the first dimension
      - columns: the second dimension
  */
  func dispatch(pipeline: MTLComputePipelineState, rows: Int, columns: Int) {
    let h = pipeline.threadExecutionWidth
    let w = pipeline.maxTotalThreadsPerThreadgroup / h

    let threadGroupSize = MTLSizeMake(w, h, 1)

    let threadGroups = MTLSizeMake(
      (rows    + threadGroupSize.width  - 1) / threadGroupSize.width,
      (columns + threadGroupSize.height - 1) / threadGroupSize.height, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}
