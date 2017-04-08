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
    Sets the parameters of the command encoder with less boilerplate.
    
    - Parameters:
      - parameters: an array that may contain MTLBuffer or MTLTexture objects, 
        or types such as UInt32 or structs
  */
  func configure(parameters: [Any]) {
    for i in 0..<parameters.count {
      var obj = parameters[i]
      if let buffer = obj as? MTLBuffer {
        setBuffer(buffer, offset: 0, at: i)
      } else if let texture = obj as? MTLTexture {
        setTexture(texture, at: i)
      } else {
        setBytes(&obj, length: MemoryLayout.size(ofValue: obj), at: i)
      }
    }
  }

  /**
    Dispatches a compute kernel on a 1-dimensional grid.
    
    - Parameters:
      - count: the number of elements to process
  */
  func dispatch(pipeline: MTLComputePipelineState, count: Int) {
    // Round off count to the nearest multiple of threadExecutionWidth.
    let width = pipeline.threadExecutionWidth
    let rounded = ((count + width - 1) / width) * width

    let blockSize = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
    let numBlocks = (count + blockSize - 1) / blockSize

    let threadGroupSize = MTLSizeMake(blockSize, 1, 1)
    let threadGroups = MTLSizeMake(numBlocks, 1, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }

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

  /**
    Dispatches a compute kernel on an MPSImage's texture or texture array.
  */
  func dispatch(pipeline: MTLComputePipelineState, image: MPSImage) {
    let numSlices = ((image.featureChannels + 3)/4) * image.numberOfImages

    let h, w, d: Int
    if numSlices == 1 {
      h = pipeline.threadExecutionWidth
      w = pipeline.maxTotalThreadsPerThreadgroup / h
      d = 1
    } else {
      // NOTE: I have no idea if this is the best way to divide up the work.
      // Maybe it doesn't make much sense to work on 2 or more slices at once.
      // Need to measure this!
      h = 16; w = 16; d = 2
    }

    let threadGroupSize = MTLSizeMake(w, h, d)
    let threadGroups = MTLSizeMake(
      (image.width  + threadGroupSize.width  - 1) / threadGroupSize.width,
      (image.height + threadGroupSize.height - 1) / threadGroupSize.height,
      (numSlices    + threadGroupSize.depth  - 1) / threadGroupSize.depth)
    
    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}
