#include <metal_stdlib>
using namespace metal;

struct Vertex {
  float4 position [[position]];
  float2 texCoord;
};

struct Uniforms {
  float4x4 modelMatrix;
};

vertex Vertex vertexFunc(
  const device Vertex *vertices [[buffer(0)]],
  constant Uniforms &uniforms [[buffer(1)]],
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

kernel void adjust_mean_bgr(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  half4 inColor = inTexture.read(gid);
  half4 outColor = half4(inColor.x*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.z*255.0 - 123.68, 1.0);
  outTexture.write(outColor, gid);
}
