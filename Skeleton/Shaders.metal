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
  texture2d_array<half> texture [[texture(0)]])
{
  constexpr sampler s;
  half4 inColor = texture.sample(s, vert.texCoord, uniforms.textureIndex);

  half4 outColor;
  if (uniforms.channelIndex == -1) {
    outColor = inColor;
  } else if (uniforms.channelIndex == 0) {
    outColor = half4(inColor.x, inColor.x, inColor.x, 1.0h);
  } else if (uniforms.channelIndex == 1) {
    outColor = half4(inColor.y, inColor.y, inColor.y, 1.0h);
  } else if (uniforms.channelIndex == 2) {
    outColor = half4(inColor.z, inColor.z, inColor.z, 1.0h);
  } else {
    outColor = half4(inColor.w, inColor.w, inColor.w, 1.0h);
  }
  return outColor;
};

kernel void adjust_mean_bgr(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  half4 inColor = inTexture.read(gid);
  half4 outColor = half4(inColor.x*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.z*255.0h - 123.68h, 1.0h);
  outTexture.write(outColor, gid);
}
