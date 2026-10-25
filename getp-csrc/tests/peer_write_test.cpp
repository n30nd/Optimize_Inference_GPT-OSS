// Checks an assumption of the paired-GCD fast path: a GCD sees the data its
// partner wrote into its (coarse-grained) memory over the link, even when its
// own L2 cache still holds the old lines. The engine relies on this for
// t16_peer (written by the partner's add_rmsnorm_bf16, read by the gather).
// Same steps as the engine: B reads X (the lines land in B's L2), A writes X
// in B's memory, A signals (release, system scope), B waits (acquire) and
// checks every value of X. ep_signal_kernel / ep_wait_kernel are copies of
// the ones in kernels/moe_ep.hpp.
//
// Build from the repo root:
//   hipcc --std=c++17 -O3 --offload-arch=gfx90a getp-csrc/tests/peer_write_test.cpp -o peer_write_test
// Run on the two GCDs of one package:
//   tools/gpulock.sh 0,1 -- ./peer_write_test [iterations=3000] [ints=262144]
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s line %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

__global__ void ep_signal_kernel(int *flag, int value) {
  __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void ep_wait_kernel(const int *flag, int value) {
  while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value)
    __builtin_amdgcn_s_sleep(2);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}
__global__ void read_sum(const int *X, int n, int *sink) {
  int s = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += X[i];
  if (s == 0x7fffffff) sink[0] = s;
}
__global__ void write_val(int *X, int n, int v) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) X[i] = v;
}
__global__ void check_val(const int *X, int n, int v, int *bad) {
  int b = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) b += X[i] != v;
  if (b) atomicAdd(bad, b);
}

int main(int argc, char **argv) {
  const int A = 0, B = 1, iters = argc > 1 ? atoi(argv[1]) : 3000;
  const int n = (argc > 2 ? atoi(argv[2]) : 1 << 18); // ints (1 MB default): fits in the 8 MB L2
  for (auto [x, y] : {std::make_pair(A, B), std::make_pair(B, A)}) {
    CK(hipSetDevice(x));
    hipError_t e = hipDeviceEnablePeerAccess(y, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
  }
  int *X, *bad, *sink, *flagA, *flagB;
  hipStream_t sA, sB;
  CK(hipSetDevice(B));
  CK(hipMalloc(&X, n * 4)); CK(hipMemset(X, 0, n * 4));
  CK(hipMalloc(&bad, 4)); CK(hipMemset(bad, 0, 4)); CK(hipMalloc(&sink, 4));
  CK(hipExtMallocWithFlags((void **)&flagB, 64, hipDeviceMallocFinegrained)); CK(hipMemset(flagB, 0, 64));
  CK(hipStreamCreate(&sB));
  CK(hipSetDevice(A));
  CK(hipExtMallocWithFlags((void **)&flagA, 64, hipDeviceMallocFinegrained)); CK(hipMemset(flagA, 0, 64));
  CK(hipStreamCreate(&sA));
  CK(hipSetDevice(B)); CK(hipDeviceSynchronize());
  CK(hipSetDevice(A)); CK(hipDeviceSynchronize());
  for (int it = 1; it <= iters; it++) {
    CK(hipSetDevice(B));
    read_sum<<<104, 256, 0, sB>>>(X, n, sink);          // X (value it-1) now in B's L2
    ep_signal_kernel<<<1, 1, 0, sB>>>(flagB, it);        // "I have read"
    CK(hipSetDevice(A));
    ep_wait_kernel<<<1, 1, 0, sA>>>(flagB, it);
    write_val<<<104, 256, 0, sA>>>(X, n, it);            // remote write into B's memory
    ep_signal_kernel<<<1, 1, 0, sA>>>(flagA, it);
    CK(hipSetDevice(B));
    ep_wait_kernel<<<1, 1, 0, sB>>>(flagA, it);
    check_val<<<104, 256, 0, sB>>>(X, n, it, bad);
  }
  CK(hipSetDevice(A)); CK(hipStreamSynchronize(sA));
  CK(hipSetDevice(B)); CK(hipStreamSynchronize(sB));
  int h = 0; CK(hipMemcpy(&h, bad, 4, hipMemcpyDeviceToHost));
  printf("iters %d, %d ints (%d KB): stale values seen %d\n", iters, n, n / 256, h);
  return h != 0;
}
