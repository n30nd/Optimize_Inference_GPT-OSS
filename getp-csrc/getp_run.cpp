// Unified getp_run.cpp that dispatches to model-specific implementations.
// The runtime automatically selects the correct backend (20B or 120B)
// based on the loaded transformer configuration, with an optional
// override via the GETP_MODEL_VARIANT environment variable.

#ifndef GETP_RUN
#define GETP_RUN

#include "getp_eval.cpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "getp_run_20b.cpp"
#include "getp_run_120b.cpp"

namespace {

using WarmUpFn = void (*)(Transformer *, Tokenizer *);
using InferenceFn = long long (*)(Transformer *, Tokenizer *, Sampler *,
                                  Requests *);
using FinishFn = void (*)(Transformer *, Tokenizer *);

enum class ModelVariant {
  kUnknown = 0,
  k20B,
  k120B,
};

const char *variant_name(ModelVariant variant) {
  switch (variant) {
  case ModelVariant::k20B:
    return "20B";
  case ModelVariant::k120B:
    return "120B";
  default:
    return "Unknown";
  }
}

bool equals_ignore_case(const char *lhs, const char *rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return false;
  }
  while (*lhs && *rhs) {
    const unsigned char cl = static_cast<unsigned char>(*lhs);
    const unsigned char cr = static_cast<unsigned char>(*rhs);
    if (std::tolower(cl) != std::tolower(cr)) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

ModelVariant read_env_override() {
  const char *env = std::getenv("GETP_MODEL_VARIANT");
  if (!env || !*env) {
    return ModelVariant::kUnknown;
  }
  if (equals_ignore_case(env, "20") || equals_ignore_case(env, "20B")) {
    return ModelVariant::k20B;
  }
  if (equals_ignore_case(env, "120") || equals_ignore_case(env, "120B")) {
    return ModelVariant::k120B;
  }
  std::fprintf(stderr,
               "[getp] Warning: unrecognized GETP_MODEL_VARIANT='%s'. Expected "
               "20B or 120B.\n",
               env);
  std::fflush(stderr);
  return ModelVariant::kUnknown;
}

ModelVariant deduce_from_config(const Config &config) {
  switch (config.n_layers) {
  case 24:
    return ModelVariant::k20B;
  case 36:
    return ModelVariant::k120B;
  default:
    break;
  }
  // Heuristic fallback if layer count is not the canonical value.
  if (config.hidden_dim > 7000) {
    return ModelVariant::k120B;
  }
  if (config.hidden_dim > 0 && config.hidden_dim <= 7000) {
    return ModelVariant::k20B;
  }
  return ModelVariant::kUnknown;
}

std::mutex g_variant_mutex;
ModelVariant g_selected_variant = ModelVariant::kUnknown;
bool g_variant_initialized = false;
WarmUpFn g_warmup_impl = nullptr;
InferenceFn g_inference_impl = nullptr;
FinishFn g_finish_impl = nullptr;

void bind_variant(ModelVariant variant) {
  switch (variant) {
  case ModelVariant::k20B:
    g_warmup_impl = &getp_variants::variant20b::warm_up;
    g_inference_impl = &getp_variants::variant20b::inference;
    g_finish_impl = &getp_variants::variant20b::finish;
    break;
  case ModelVariant::k120B:
    g_warmup_impl = &getp_variants::variant120b::warm_up;
    g_inference_impl = &getp_variants::variant120b::inference;
    g_finish_impl = &getp_variants::variant120b::finish;
    break;
  default:
    g_warmup_impl = nullptr;
    g_inference_impl = nullptr;
    g_finish_impl = nullptr;
    break;
  }
}

ModelVariant ensure_variant_initialized(const Config &config, bool verbose) {
  std::lock_guard<std::mutex> guard(g_variant_mutex);
  if (!g_variant_initialized) {
    ModelVariant variant = read_env_override();
    ModelVariant inferred = deduce_from_config(config);
    if (variant == ModelVariant::kUnknown) {
      variant = inferred;
    } else if (inferred != ModelVariant::kUnknown && inferred != variant) {
      if (verbose) {
        std::fprintf(stderr,
                     "[getp] GETP_MODEL_VARIANT=%s overrides configuration that "
                     "maps to %s (n_layers=%d).\n",
                     variant_name(variant), variant_name(inferred),
                     config.n_layers);
        std::fflush(stderr);
      }
    }
    if (variant == ModelVariant::kUnknown) {
      std::fprintf(stderr,
                   "[getp] Unable to determine model variant (n_layers=%d, "
                   "hidden_dim=%d). Set GETP_MODEL_VARIANT=20B or 120B.\n",
                   config.n_layers, config.hidden_dim);
      std::fflush(stderr);
      std::exit(EXIT_FAILURE);
    }
    g_selected_variant = variant;
    bind_variant(variant);
    g_variant_initialized = true;
    if (verbose) {
      std::printf("[getp] Selected model variant: %s (n_layers=%d)\n",
                  variant_name(g_selected_variant), config.n_layers);
      std::fflush(stdout);
    }
  } else {
    ModelVariant inferred = deduce_from_config(config);
    if (inferred != ModelVariant::kUnknown && inferred != g_selected_variant) {
      std::fprintf(stderr,
                   "[getp] Configuration (n_layers=%d) resolves to %s, but runtime "
                   "is locked to %s. Call finish() before switching variants.\n",
                   config.n_layers, variant_name(inferred),
                   variant_name(g_selected_variant));
      std::fflush(stderr);
      std::exit(EXIT_FAILURE);
    }
    if (!g_warmup_impl || !g_inference_impl || !g_finish_impl) {
      bind_variant(g_selected_variant);
    }
  }
  return g_selected_variant;
}

void clear_variant_selection() {
  std::lock_guard<std::mutex> guard(g_variant_mutex);
  g_variant_initialized = false;
  g_selected_variant = ModelVariant::kUnknown;
  g_warmup_impl = nullptr;
  g_inference_impl = nullptr;
  g_finish_impl = nullptr;
}

} // namespace

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  if (transformer == nullptr) {
    std::fprintf(stderr, "[getp] warm_up received null transformer pointer.\n");
    std::fflush(stderr);
    return;
  }
  ensure_variant_initialized(transformer->config, true);
  WarmUpFn fn = g_warmup_impl;
  if (fn) {
    fn(transformer, tokenizer);
  }
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (!transformer) {
    return 0;
  }
  InferenceFn fn = g_inference_impl;
  if (!fn) {
    ensure_variant_initialized(transformer->config, false);
    fn = g_inference_impl;
    if (!fn) {
      return 0;
    }
  }
  return fn(transformer, tokenizer, sampler, requests);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  if (!transformer) {
    clear_variant_selection();
    return;
  }
  FinishFn fn = g_finish_impl;
  if (!fn) {
    ensure_variant_initialized(transformer->config, false);
    fn = g_finish_impl;
    if (!fn) {
      clear_variant_selection();
      return;
    }
  }
  fn(transformer, tokenizer);
  clear_variant_selection();
}

#endif // GETP_RUN
