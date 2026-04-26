// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "GGML/core/ggml_core.h"
#include "wasinn_bitnet.h"
#include "wasinn_chattts.h"
#include "wasinn_mlx.h"
#include "wasinn_neuralspeed.h"
#include "wasinn_onnx.h"
#include "wasinn_openvino.h"
#include "wasinn_openvino_genai.h"
#include "wasinn_piper.h"
#include "wasinn_tf.h"
#include "wasinn_tfl.h"
#include "wasinn_torch.h"
#include "wasinn_whisper.h"
#include "wasinntypes.h"

#include "common/errcode.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace WasmEdge {
namespace Host {
namespace WASINN {

namespace detail {
template <typename T, typename V> struct VariantIndex;

template <typename T, typename... Types>
struct VariantIndex<T, std::variant<T, Types...>>
    : std::integral_constant<size_t, 0> {};

template <typename T, typename H, typename... Types>
struct VariantIndex<T, std::variant<H, Types...>>
    : std::integral_constant<
          std::size_t, VariantIndex<T, std::variant<Types...>>::value + 1> {};

template <typename T, typename V>
inline constexpr std::size_t VariantIndexV = VariantIndex<T, V>::value;

template <Backend B> struct BackendTrait;
#define EACH(B)                                                                \
  template <> struct BackendTrait<Backend::B> {                                \
    using Graph = B::Graph;                                                    \
    using Context = B::Context;                                                \
  };
FOR_EACH_BACKEND(EACH)
#undef EACH

template <Backend B> using BackendGraphT = typename BackendTrait<B>::Graph;
template <Backend B> using BackendContextT = typename BackendTrait<B>::Context;
} // namespace detail

class Graph {
public:
  Graph() = delete;
  Graph(Backend BE) noexcept : Impl(std::in_place_type_t<std::monostate>()) {
    init(BE);
  }

  Backend getBackend() const noexcept {
    using V = std::decay_t<decltype(Impl)>;
    switch (Impl.index()) {
#define EACH(B)                                                                \
  case detail::VariantIndexV<B::Graph, V>:                                     \
    return Backend::B;
      FOR_EACH_BACKEND(EACH)
#undef EACH
    default:
      __builtin_unreachable();
    }
  }

  template <Backend B> auto &get() noexcept {
    return *std::get_if<detail::BackendGraphT<B>>(&Impl);
  }
  template <Backend B> const auto &get() const noexcept {
    return *std::get_if<detail::BackendGraphT<B>>(&Impl);
  }
  template <typename T> auto &get() noexcept { return *std::get_if<T>(&Impl); }
  template <typename T> const auto &get() const noexcept {
    return *std::get_if<T>(&Impl);
  }

  void init(Backend BE) noexcept {
    switch (BE) {
#define EACH(B)                                                                \
  case Backend::B:                                                             \
    Impl.emplace<B::Graph>();                                                  \
    break;
      FOR_EACH_BACKEND(EACH)
#undef EACH
    default:
      __builtin_unreachable();
    }
    Stat = Status::Uninitialized;
    CtxCnt = 0;
  }
  void reset() noexcept {
    Impl = std::monostate{};
    Stat = Status::Uninitialized;
    CtxCnt = 0;
  }
  void increaseContext() noexcept { CtxCnt++; }
  void decreaseContext() noexcept {
    assuming(CtxCnt > 0);
    CtxCnt--;
  }
  uint32_t getContextCount() const noexcept { return CtxCnt; }
  bool isFinalized() const noexcept {
    return Stat == Status::Uninitialized || Stat == Status::Finalized;
  }
  bool isReady() const noexcept { return Stat == Status::Ready; }
  void setInvalid() noexcept { Stat = Status::Invalid; }
  void setFinalized() noexcept { Stat = Status::Finalized; }
  void setReady() noexcept { Stat = Status::Ready; }

private:
  std::variant<
#define EACH(B) B::Graph,
      FOR_EACH_BACKEND(EACH)
#undef EACH
          std::monostate>
      Impl;
  enum class Status : uint8_t { Uninitialized, Invalid, Finalized, Ready };
  Status Stat;
  uint32_t CtxCnt;
};

class Context {
public:
  Context() = delete;
  Context(uint32_t GId, Graph &G) noexcept
      : Impl(std::in_place_type_t<std::monostate>()) {
    init(GId, G);
  }

  Backend getBackend() const noexcept {
    using V = std::decay_t<decltype(Impl)>;
    switch (Impl.index()) {
#define EACH(B)                                                                \
  case detail::VariantIndexV<B::Context, V>:                                   \
    return Backend::B;
      FOR_EACH_BACKEND(EACH)
#undef EACH
    default:
      __builtin_unreachable();
    }
  }

  template <Backend B> auto &get() noexcept {
    return *std::get_if<detail::BackendContextT<B>>(&Impl);
  }
  template <Backend B> const auto &get() const noexcept {
    return *std::get_if<detail::BackendContextT<B>>(&Impl);
  }
  template <typename T> auto &get() noexcept { return *std::get_if<T>(&Impl); }
  template <typename T> const auto &get() const noexcept {
    return *std::get_if<T>(&Impl);
  }

  void init(uint32_t GId, Graph &G) noexcept {
    switch (G.getBackend()) {
#define EACH(B)                                                                \
  case Backend::B:                                                             \
    Impl.emplace<B::Context>(GId, G.get<Backend::B>());                        \
    break;
      FOR_EACH_BACKEND(EACH)
#undef EACH
    default:
      __builtin_unreachable();
    }
    Stat = Status::Uninitialized;
    GraphId = GId;
  }
  void reset() noexcept {
    Impl = std::monostate{};
    Stat = Status::Uninitialized;
    GraphId = 0;
  }
  uint32_t getGraphId() const noexcept {
    return static_cast<uint32_t>(GraphId);
  }
  bool isReady() const noexcept { return Stat == Status::Ready; }
  void setReady() noexcept { Stat = Status::Ready; }

private:
  std::variant<
#define EACH(B) B::Context,
      FOR_EACH_BACKEND(EACH)
#undef EACH
          std::monostate>
      Impl;
  enum class Status : uint8_t { Uninitialized, Ready };
  Status Stat;
  uint32_t GraphId;
};

class GraphStore {
public:
  static constexpr uint32_t InvalidId = std::numeric_limits<uint32_t>::max();

  uint32_t newGraph(Backend BE) noexcept;
  uint32_t newContext(uint32_t GId) noexcept;
  bool hasGraph(uint32_t Id) noexcept;
  bool isGraphReady(uint32_t Id) noexcept;
  bool isGraphFinalized(uint32_t Id) noexcept;
  std::optional<Backend> getGraphBackend(uint32_t Id) noexcept;
  bool hasContext(uint32_t Id) noexcept;
  bool isContextReady(uint32_t Id) noexcept;
  std::optional<Backend> getContextBackend(uint32_t Id) noexcept;
  std::optional<uint32_t> getContextGraphId(uint32_t Id) noexcept;
  bool isContextGraphReady(uint32_t Id) noexcept;
  void setGraphReady(uint32_t Id) noexcept;
  void setGraphInvalid(uint32_t Id) noexcept;
  void setContextGraphReady(uint32_t Id) noexcept;
  void setContextGraphInvalid(uint32_t Id) noexcept;
  void deleteGraph(uint32_t Id) noexcept;
  void deleteContext(uint32_t Id) noexcept;
  void setContextReady(uint32_t Id) noexcept;
  Graph *getGraph(uint32_t Id) noexcept;
  Context *getContext(uint32_t Id) noexcept;
  Context *getReadyContext(uint32_t Id) noexcept;

#ifdef WASMEDGE_WASI_NN_TESTING
  void swapGraphContextForTesting(std::deque<Graph> &Graphs,
                                  std::deque<Context> &Contexts) noexcept {
    NNGraph.swap(Graphs);
    NNContext.swap(Contexts);
  }
#endif

private:
  std::deque<Graph> NNGraph;
  std::deque<Context> NNContext;
  std::shared_mutex GraphMutex;
  std::unordered_set<uint32_t> NNGraphRecycle;
  std::unordered_set<uint32_t> NNContextRecycle;
};

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
