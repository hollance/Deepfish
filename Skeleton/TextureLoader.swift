import MetalKit

let textureLoader: MTKTextureLoader = {
  return MTKTextureLoader(device: MTLCreateSystemDefaultDevice()!)
}()

func loadTexture(named filename: String) -> MTLTexture? {
  if let url = Bundle.main.url(forResource: filename, withExtension: "") {
    return loadTexture(url: url)
  } else {
    print("Error: could not find image \(filename)")
    return nil
  }
}

func loadTexture(url: URL) -> MTLTexture? {
  do {
    // Note: the SRGB option should be set to false, otherwise the image
    // appears way too dark, since it wasn't actually saved as SRGB.
    return try textureLoader.newTexture(withContentsOf: url, options: [
      MTKTextureLoaderOptionSRGB : NSNumber(value: false)
    ])
  } catch {
    print("Error: could not load texture \(error)")
    return nil
  }
}
