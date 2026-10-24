// Minimal JSON reader: enough for safetensors headers and config.json files.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

struct JsonValue {
  enum Type { Null, Bool, Number, String, Array, Object };
  Type type = Null;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> arr;
  std::vector<std::pair<std::string, JsonValue>> obj;

  const JsonValue *get(const char *key) const {
    if (type != Object)
      return NULL;
    for (const auto &kv : obj)
      if (kv.first == key)
        return &kv.second;
    return NULL;
  }
};

class JsonParser {
public:
  JsonParser(const char *text, size_t len) : p_(text), end_(text + len) {}

  bool parse(JsonValue *out) {
    skip_ws();
    if (!parse_value(out))
      return false;
    skip_ws();
    return p_ == end_;
  }

private:
  const char *p_;
  const char *end_;

  void skip_ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t'))
      p_++;
  }

  bool literal(const char *word) {
    size_t n = strlen(word);
    if ((size_t)(end_ - p_) < n || strncmp(p_, word, n) != 0)
      return false;
    p_ += n;
    return true;
  }

  static void append_utf8(std::string *s, unsigned cp) {
    if (cp < 0x80) {
      s->push_back((char)cp);
    } else if (cp < 0x800) {
      s->push_back((char)(0xC0 | (cp >> 6)));
      s->push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      s->push_back((char)(0xE0 | (cp >> 12)));
      s->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      s->push_back((char)(0x80 | (cp & 0x3F)));
    } else {
      s->push_back((char)(0xF0 | (cp >> 18)));
      s->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
      s->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      s->push_back((char)(0x80 | (cp & 0x3F)));
    }
  }

  bool parse_hex4(unsigned *cp) {
    if (end_ - p_ < 4)
      return false;
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
      char c = *p_++;
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= c - '0';
      else if (c >= 'a' && c <= 'f')
        v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v |= c - 'A' + 10;
      else
        return false;
    }
    *cp = v;
    return true;
  }

  bool parse_string(std::string *s) {
    if (p_ >= end_ || *p_ != '"')
      return false;
    p_++;
    while (p_ < end_ && *p_ != '"') {
      char c = *p_++;
      if (c != '\\') {
        s->push_back(c);
        continue;
      }
      if (p_ >= end_)
        return false;
      char e = *p_++;
      switch (e) {
      case '"': s->push_back('"'); break;
      case '\\': s->push_back('\\'); break;
      case '/': s->push_back('/'); break;
      case 'b': s->push_back('\b'); break;
      case 'f': s->push_back('\f'); break;
      case 'n': s->push_back('\n'); break;
      case 'r': s->push_back('\r'); break;
      case 't': s->push_back('\t'); break;
      case 'u': {
        unsigned cp;
        if (!parse_hex4(&cp))
          return false;
        // surrogate pair
        if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' &&
            p_[1] == 'u') {
          p_ += 2;
          unsigned lo;
          if (!parse_hex4(&lo))
            return false;
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        }
        append_utf8(s, cp);
        break;
      }
      default:
        return false;
      }
    }
    if (p_ >= end_)
      return false;
    p_++; // closing quote
    return true;
  }

  bool parse_number(double *v) {
    char buf[64];
    size_t n = 0;
    while (p_ < end_ && n < sizeof(buf) - 1 &&
           (strchr("+-0123456789.eE", *p_) != NULL))
      buf[n++] = *p_++;
    buf[n] = '\0';
    if (n == 0)
      return false;
    char *stop;
    *v = strtod(buf, &stop);
    return *stop == '\0';
  }

  bool parse_value(JsonValue *v) {
    skip_ws();
    if (p_ >= end_)
      return false;
    char c = *p_;
    if (c == '{') {
      v->type = JsonValue::Object;
      p_++;
      skip_ws();
      if (p_ < end_ && *p_ == '}') {
        p_++;
        return true;
      }
      while (true) {
        skip_ws();
        std::pair<std::string, JsonValue> kv;
        if (!parse_string(&kv.first))
          return false;
        skip_ws();
        if (p_ >= end_ || *p_ != ':')
          return false;
        p_++;
        if (!parse_value(&kv.second))
          return false;
        v->obj.push_back(std::move(kv));
        skip_ws();
        if (p_ < end_ && *p_ == ',') {
          p_++;
          continue;
        }
        if (p_ < end_ && *p_ == '}') {
          p_++;
          return true;
        }
        return false;
      }
    }
    if (c == '[') {
      v->type = JsonValue::Array;
      p_++;
      skip_ws();
      if (p_ < end_ && *p_ == ']') {
        p_++;
        return true;
      }
      while (true) {
        JsonValue item;
        if (!parse_value(&item))
          return false;
        v->arr.push_back(std::move(item));
        skip_ws();
        if (p_ < end_ && *p_ == ',') {
          p_++;
          continue;
        }
        if (p_ < end_ && *p_ == ']') {
          p_++;
          return true;
        }
        return false;
      }
    }
    if (c == '"') {
      v->type = JsonValue::String;
      return parse_string(&v->str);
    }
    if (literal("true")) {
      v->type = JsonValue::Bool;
      v->boolean = true;
      return true;
    }
    if (literal("false")) {
      v->type = JsonValue::Bool;
      return true;
    }
    if (literal("null")) {
      v->type = JsonValue::Null;
      return true;
    }
    v->type = JsonValue::Number;
    return parse_number(&v->number);
  }
};

static bool json_parse(const char *text, size_t len, JsonValue *out) {
  JsonParser parser(text, len);
  return parser.parse(out);
}

static bool json_parse_file(const char *path, JsonValue *out) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return false;
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
    text.append(buf, n);
  fclose(f);
  return json_parse(text.data(), text.size(), out);
}
