/// Zero-sized handle for the AVX2 SIMD backend.
///
/// Used as the dispatch target for `KernelDispatch<K>` impls in the `launch/`
/// submodules.  No device initialisation is needed, AVX2 availability is
/// checked at runtime via `is_x86_feature_detected!("avx2")`.
pub struct Avx2Device;
