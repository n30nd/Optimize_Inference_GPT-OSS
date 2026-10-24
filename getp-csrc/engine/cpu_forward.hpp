// CPU forward pass over Hugging Face weights. Same code as forward() in
// run.cpp except the two expert matmuls, which read MXFP4 weights directly.
// Each weight decodes to the exact fp32 value the exported .bin holds and the
// arithmetic order is unchanged, so the logits are bit-identical to
// forward() on an exported .bin (checked by tests/cpu_forward_equiv.cpp).
#pragma once

#include "hf_model.hpp"

// xout (d,) = W (d, n) @ x (n,), W in MXFP4. Mirrors matmul() in run.cpp.
static void matmul_mxfp4(float *xout, float *x, const uint8_t *blocks,
                         const uint8_t *scales, int n, int d) {
  const int groups = n / 32;
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    const uint8_t *brow = blocks + 1ll * i * groups * 16;
    const uint8_t *srow = scales + 1ll * i * groups;
    double val = 0.0f;
    for (int g = 0; g < groups; g++) {
      float scale = ldexpf(1.0f, (int)srow[g] - 127);
      for (int k = 0; k < 16; k++) {
        uint8_t byte = brow[g * 16 + k];
        float w0 = kFp4Values[byte & 0x0F] * scale;
        float w1 = kFp4Values[byte >> 4] * scale;
        val += w0 * x[g * 32 + 2 * k];
        val += w1 * x[g * 32 + 2 * k + 1];
      }
    }
    xout[i] = val;
  }
}

static float *hf_cpu_forward(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  float *x = s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + 1ll * token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

    int loff = l * p->seq_len * kv_dim;
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                  (head_dim * p->n_attn_heads +
                                   2 * head_dim * p->n_kv_heads);
    float *b_qkv =
        w->b_qkv +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    matmul(s->qkv, s->t, w_qkv, hidden_dim,
           (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
      s->qkv[i] += b_qkv[i];
    }
    memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float));
    memcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
           head_dim * p->n_kv_heads * sizeof(float));
    memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
           head_dim * p->n_kv_heads * sizeof(float));

    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    float *cos_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    float *sin_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                    p->initial_context_length, ntk_beta, ntk_alpha, cos_vals,
                    sin_vals);
    apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);
    free(cos_vals);
    free(sin_vals);

    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_attn_heads; h++) {
      float *q = s->q + h * head_dim;
      float *att = s->att + h * p->seq_len;
      for (int t = 0; t <= pos; t++) {
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        double score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_dim);
        if (p->sliding_window > 0 && (l % 2 == 0)) {
          score += s->mask[pos * p->seq_len + t];
        }
        att[t] = score;
      }
      att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
      softmax(att, pos + 2);

      float *tb = s->tb + h * head_dim;
      memset(tb, 0, head_dim * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        float a = att[t];
        for (int i = 0; i < head_dim; i++) {
          tb[i] += a * v[i];
        }
      }
    }
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;
    matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_o[i];
    }
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->tb2[i];
    }

    rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

    float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    float *b_router = w->b_router + 1ll * l * n_experts;
    matmul(s->router_score, s->t, w_router, hidden_dim, n_experts);
    for (int i = 0; i < n_experts; i++) {
      s->router_score[i] += b_router[i];
    }
    topk(s->topk_v, s->topk_i, s->router_score, n_experts,
         p->experts_per_token);
    softmax(s->topk_v, p->experts_per_token);

    const HfLayerExperts &ex = g_hf.experts[l];
    const long long mlp1_rows = 2ll * intermediate_dim;
    memset(s->e_agg, 0, hidden_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
      float expert_w = 0;
      int in_topk = 0;
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        if (s->topk_i[idx] == e) {
          in_topk = 1;
          expert_w = s->topk_v[idx];
          break;
        }
      }

      if (in_topk) {
        matmul_mxfp4(s->mlp1_out, s->t,
                     ex.mlp1_blocks + e * mlp1_rows * (hidden_dim / 32) * 16,
                     ex.mlp1_scales + e * mlp1_rows * (hidden_dim / 32),
                     hidden_dim, 2 * intermediate_dim);
        float *b_mlp1 =
            w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
        for (int i = 0; i < 2 * p->intermediate_dim; i++) {
          s->mlp1_out[i] += b_mlp1[i];
        }
        for (int j = 0; j < p->intermediate_dim; j++) {
          s->gate[j] = s->mlp1_out[2 * j];
          s->up[j] = s->mlp1_out[2 * j + 1];
        }

        const float alpha = 1.702f;
        for (int i = 0; i < p->intermediate_dim; i++) {
          float val = s->gate[i];
          float up_val = s->up[i];
          if (val > p->swiglu_limit)
            val = p->swiglu_limit;
          if (up_val > p->swiglu_limit)
            up_val = p->swiglu_limit;
          if (up_val < -p->swiglu_limit)
            up_val = -p->swiglu_limit;
          val *= (1.0f / (1.0f + expf(-alpha * val)));
          val *= (up_val + 1.0f);
          s->gate_up[i] = val;
        }

        matmul_mxfp4(s->tb2, s->gate_up,
                     ex.mlp2_blocks + 1ll * e * hidden_dim * (intermediate_dim / 32) * 16,
                     ex.mlp2_scales + 1ll * e * hidden_dim * (intermediate_dim / 32),
                     intermediate_dim, hidden_dim);
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
          s->tb2[i] += b_mlp2[i];
        }

        for (int i = 0; i < hidden_dim; i++) {
          s->e_agg[i] += s->tb2[i] * expert_w;
        }
      }
    }

    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->e_agg[i];
    }
  }
  rmsnorm(x, x, w->rms_out_w, hidden_dim);

  matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
  return s->logits;
}
