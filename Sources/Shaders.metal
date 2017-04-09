#include <metal_stdlib>
using namespace metal;

struct Vertex {
  float4 position [[position]];
  float2 texCoord;
};

struct VertexUniforms {
  float4x4 modelMatrix;
};

struct FragmentUniforms {
  short textureIndex;
  short channelIndex;
};

vertex Vertex vertexFunc(
  const device Vertex *vertices [[buffer(0)]],
  constant VertexUniforms &uniforms [[buffer(1)]],
  uint vid [[vertex_id]])
{
  Vertex vert;
  vert.position = uniforms.modelMatrix * float4(vertices[vid].position);
  vert.texCoord = vertices[vid].texCoord;
  return vert;
};

fragment half4 fragmentFunc(
  Vertex vert [[stage_in]],
  texture2d<half> texture [[texture(0)]])
{
  constexpr sampler s;
  return texture.sample(s, vert.texCoord);
};

fragment half4 fragmentFuncA(
  Vertex vert [[stage_in]],
  constant FragmentUniforms &uniforms [[buffer(0)]],
  texture2d_array<half> texture [[texture(0)]],
  texture2d_array<half> maxTexture [[texture(1)]])
{
  constexpr sampler s;
  half4 inColor = texture.sample(s, vert.texCoord, uniforms.textureIndex);
  half4 maxValue = maxTexture.sample(s, float2(0, 0), uniforms.textureIndex);

  // We read from a single channel in the texture and convert that to a
  // grayscale color. To normalize this to a color in the range [0,1], we
  // divide the activation value by the maximum value from that channel.

  half4 outColor;
  if (uniforms.channelIndex == -1) {   // pass through as RGBA
    outColor = inColor / maxValue;
  } else if (uniforms.channelIndex == 0) {
    outColor = half4(inColor.x, inColor.x, inColor.x, 1.0h);
    outColor /= half4(maxValue.x, maxValue.x, maxValue.x, 1.0h);
  } else if (uniforms.channelIndex == 1) {
    outColor = half4(inColor.y, inColor.y, inColor.y, 1.0h);
    outColor /= half4(maxValue.y, maxValue.y, maxValue.y, 1.0h);
  } else if (uniforms.channelIndex == 2) {
    outColor = half4(inColor.z, inColor.z, inColor.z, 1.0h);
    outColor /= half4(maxValue.z, maxValue.z, maxValue.z, 1.0h);
  } else {
    outColor = half4(inColor.w, inColor.w, inColor.w, 1.0h);
    outColor /= half4(maxValue.w, maxValue.w, maxValue.w, 1.0h);
  }
  return outColor;
};


kernel void adjust_mean_rgb(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.z*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.x*255.0 - 123.68, 0.0);
  outTexture.write(outColor, gid);
}

kernel void adjust_mean_bgr(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  half4 inColor = inTexture.read(gid);
  half4 outColor = half4(inColor.x*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.z*255.0h - 123.68h, 1.0h);
  outTexture.write(outColor, gid);
}
