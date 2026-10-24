// Reads .safetensors files: an 8-byte little-endian header length, a JSON
// header {name: {dtype, shape, data_offsets}}, then the raw tensor bytes.
// Files are memory-mapped read-only; tensors point straight into the mapping.
#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

struct StTensor {
  std::string dtype; // "BF16", "U8", ...
  std::vector<int64_t> shape;
  const uint8_t *data = NULL;
  size_t nbytes = 0;
  int file = -1; // index into StModel::files
};

struct StFile {
  std::string path;
  int fd = -1;
  void *map = MAP_FAILED;
  size_t size = 0;
};

struct StModel {
  std::vector<StFile> files;
  std::unordered_map<std::string, StTensor> tensors;
  size_t total_bytes = 0;
};

static bool st_open_file(StModel *m, const std::string &path) {
  StFile f;
  f.path = path;
  f.fd = open(path.c_str(), O_RDONLY);
  if (f.fd < 0) {
    fprintf(stderr, "safetensors: cannot open %s\n", path.c_str());
    return false;
  }
  struct stat st;
  if (fstat(f.fd, &st) != 0 || st.st_size < 8) {
    fprintf(stderr, "safetensors: bad file %s\n", path.c_str());
    close(f.fd);
    return false;
  }
  f.size = (size_t)st.st_size;
  f.map = mmap(NULL, f.size, PROT_READ, MAP_PRIVATE, f.fd, 0);
  if (f.map == MAP_FAILED) {
    fprintf(stderr, "safetensors: mmap failed for %s\n", path.c_str());
    close(f.fd);
    return false;
  }
  const uint8_t *base = (const uint8_t *)f.map;
  uint64_t header_len = 0;
  memcpy(&header_len, base, 8);
  if (8 + header_len > f.size) {
    fprintf(stderr, "safetensors: header too long in %s\n", path.c_str());
    return false;
  }
  JsonValue header;
  if (!json_parse((const char *)base + 8, header_len, &header) ||
      header.type != JsonValue::Object) {
    fprintf(stderr, "safetensors: cannot parse header of %s\n", path.c_str());
    return false;
  }
  const uint8_t *data_start = base + 8 + header_len;
  int file_idx = (int)m->files.size();
  for (const auto &kv : header.obj) {
    if (kv.first == "__metadata__")
      continue;
    const JsonValue *dtype = kv.second.get("dtype");
    const JsonValue *shape = kv.second.get("shape");
    const JsonValue *offs = kv.second.get("data_offsets");
    if (!dtype || !shape || !offs || offs->arr.size() != 2) {
      fprintf(stderr, "safetensors: bad entry %s in %s\n", kv.first.c_str(),
              path.c_str());
      return false;
    }
    StTensor t;
    t.dtype = dtype->str;
    for (const auto &d : shape->arr)
      t.shape.push_back((int64_t)d.number);
    size_t begin = (size_t)offs->arr[0].number;
    size_t end = (size_t)offs->arr[1].number;
    if (data_start + end > base + f.size || end < begin) {
      fprintf(stderr, "safetensors: %s is out of bounds in %s\n",
              kv.first.c_str(), path.c_str());
      return false;
    }
    t.data = data_start + begin;
    t.nbytes = end - begin;
    t.file = file_idx;
    m->total_bytes += t.nbytes;
    m->tensors[kv.first] = t;
  }
  m->files.push_back(f);
  return true;
}

// Opens every *.safetensors file in `dir` (sorted by name).
static bool st_open_dir(StModel *m, const std::string &dir) {
  DIR *d = opendir(dir.c_str());
  if (!d) {
    fprintf(stderr, "safetensors: cannot open folder %s\n", dir.c_str());
    return false;
  }
  std::vector<std::string> names;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    std::string name = e->d_name;
    const std::string ext = ".safetensors";
    if (name.size() > ext.size() &&
        name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
      names.push_back(name);
  }
  closedir(d);
  std::sort(names.begin(), names.end());
  if (names.empty()) {
    fprintf(stderr, "safetensors: no .safetensors files in %s\n", dir.c_str());
    return false;
  }
  for (const auto &name : names)
    if (!st_open_file(m, dir + "/" + name))
      return false;
  return true;
}

static void st_close(StModel *m) {
  for (auto &f : m->files) {
    if (f.map != MAP_FAILED)
      munmap(f.map, f.size);
    if (f.fd >= 0)
      close(f.fd);
  }
  m->files.clear();
  m->tensors.clear();
}

// Returns the tensor or exits with a clear message if the name, dtype or
// shape is not what the caller expects.
static const StTensor &st_require(const StModel *m, const std::string &name,
                                  const char *dtype,
                                  const std::vector<int64_t> &shape) {
  auto it = m->tensors.find(name);
  if (it == m->tensors.end()) {
    fprintf(stderr, "weights: tensor %s not found\n", name.c_str());
    exit(EXIT_FAILURE);
  }
  const StTensor &t = it->second;
  if (t.dtype != dtype || t.shape != shape) {
    std::string got, want;
    for (auto v : t.shape)
      got += std::to_string(v) + ",";
    for (auto v : shape)
      want += std::to_string(v) + ",";
    fprintf(stderr, "weights: %s is %s[%s], expected %s[%s]\n", name.c_str(),
            t.dtype.c_str(), got.c_str(), dtype, want.c_str());
    exit(EXIT_FAILURE);
  }
  return t;
}
