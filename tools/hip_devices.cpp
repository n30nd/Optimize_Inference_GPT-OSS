// Prints "<hip index> <pci bus id>" for every HIP device. HIP numbering can
// differ from rocm-smi's, so tools/gpus.py joins the two on the bus id.
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) {
    fprintf(stderr, "hipGetDeviceCount failed\n");
    return 1;
  }
  for (int i = 0; i < n; i++) {
    char bus[64];
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), i) != hipSuccess) {
      fprintf(stderr, "hipDeviceGetPCIBusId(%d) failed\n", i);
      return 1;
    }
    printf("%d %s\n", i, bus);
  }
  return 0;
}
