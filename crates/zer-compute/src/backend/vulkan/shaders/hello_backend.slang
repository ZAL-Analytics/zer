// hello_backend.slang, diagnostic shader that proves Vulkan execution occurred.
// Thread (0,0,0) writes VULKAN_TOKEN to out[0]; host verifies the value.
// Vulkan is the only backend that produces this token.

static const uint VULKAN_TOKEN = 0xBE1CAFE1u;  // "BELieve CAFe Vulkan1"

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> g_out;

struct PushConstants { uint dummy; };
[[vk::push_constant]] PushConstants g_push;

[numthreads(1, 1, 1)]
void hello_backend_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0u && tid.y == 0u && tid.z == 0u) {
        g_out[0] = VULKAN_TOKEN;
    }
}
