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
