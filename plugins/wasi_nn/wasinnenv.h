// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_graph.h"
#include "wasinntypes.h"

#include "host/wasi/environ.h"

#include "common/spdlog.h"
#include "host/wasi/wasimodule.h"
#include "plugin/plugin.h"
#include "runtime/callingframe.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
#include <grpc/grpc.h>
#include <grpcpp/create_channel.h>
#endif

namespace WasmEdge {
namespace Host {
namespace WASINN {

class PreloadStore;

struct WasiNNEnvironment :
#define EACH(B) B::Environ,
    FOR_EACH_BACKEND(EACH)
#undef EACH
        std::monostate {

  using Callback = std::function<Expect<WASINN::ErrNo>(
      WASINN::WasiNNEnvironment &, Span<const Span<uint8_t>>, WASINN::Backend,
      WASINN::Device, uint32_t &)>;

  WasiNNEnvironment() noexcept;
  ~WasiNNEnvironment() noexcept;

  class GraphGuard {
  public:
    GraphGuard(WasiNNEnvironment &E, Backend BE) noexcept
        : Env(E), Id(E.newGraph(BE)) {}
    ~GraphGuard() noexcept {
      if (Active) {
        Env.deleteGraph(Id);
      }
    }
    GraphGuard(const GraphGuard &) = delete;
    GraphGuard &operator=(const GraphGuard &) = delete;
    GraphGuard(GraphGuard &&Other) noexcept
        : Env(Other.Env), Id(Other.Id), Active(Other.Active) {
      Other.Active = false;
    }
    GraphGuard &operator=(GraphGuard &&) = delete;

    uint32_t id() const noexcept { return Id; }
    template <Backend B> detail::BackendGraphT<B> &get() noexcept {
      return Env.getBackendGraph<B>(Id);
    }
    uint32_t commit() noexcept {
      Env.setGraphReady(Id);
      Active = false;
      return Id;
    }

  private:
    WasiNNEnvironment &Env;
    uint32_t Id;
    bool Active = true;
  };

  class ContextGuard {
  public:
    ContextGuard(WasiNNEnvironment &E, uint32_t GId) noexcept
        : Env(E), Id(E.newContext(GId)), Active(Id != GraphStore::InvalidId) {}
    ~ContextGuard() noexcept {
      if (Active) {
        Env.deleteContext(Id);
      }
    }
    ContextGuard(const ContextGuard &) = delete;
    ContextGuard &operator=(const ContextGuard &) = delete;
    ContextGuard(ContextGuard &&Other) noexcept
        : Env(Other.Env), Id(Other.Id), Active(Other.Active) {
      Other.Active = false;
    }
    ContextGuard &operator=(ContextGuard &&) = delete;

    uint32_t id() const noexcept { return Id; }
    bool valid() const noexcept { return Id != GraphStore::InvalidId; }
    template <Backend B> detail::BackendContextT<B> &get() noexcept {
      assuming(valid());
      return Env.getBackendContext<B>(Id);
    }
    uint32_t commit() noexcept {
      if (!valid()) {
        return Id;
      }
      Env.setContextReady(Id);
      Active = false;
      return Id;
    }

  private:
    WasiNNEnvironment &Env;
    uint32_t Id;
    bool Active = true;
  };

  bool mdGet(const std::string &Name, uint32_t &GraphId) noexcept;
  void mdRemoveById(uint32_t GraphId) noexcept;
  Expect<WASINN::ErrNo>
  mdBuild(const std::string &Name, uint32_t &GraphId, Callback Load,
          std::vector<uint8_t> Config = std::vector<uint8_t>()) noexcept;

  GraphGuard newGraphGuard(Backend BE) noexcept { return {*this, BE}; }
  ContextGuard newContextGuard(uint32_t GId) noexcept { return {*this, GId}; }
  uint32_t newReadyContext(uint32_t GId) noexcept {
    auto Context = newContextGuard(GId);
    return Context.commit();
  }
  std::unique_lock<std::shared_mutex> lockGraphOperation() const {
    return std::unique_lock<std::shared_mutex>(OperationMutex);
  }
  template <typename FuncT> decltype(auto) withGraphOperation(FuncT &&Func) {
    auto OperationLock = lockGraphOperation();
    return std::forward<FuncT>(Func)();
  }
  bool hasGraph(uint32_t Id) noexcept;
  bool isGraphReady(uint32_t Id) noexcept;
  bool isGraphFinalized(uint32_t Id) noexcept;
  std::optional<Backend> getGraphBackend(uint32_t Id) noexcept;
  bool hasContext(uint32_t Id) noexcept;
  bool isContextReady(uint32_t Id) noexcept;
  std::optional<Backend> getContextBackend(uint32_t Id) noexcept;
  std::optional<uint32_t> getContextGraphId(uint32_t Id) noexcept;
  bool isContextGraphReady(uint32_t Id) noexcept;

private:
  template <Backend B>
  detail::BackendGraphT<B> *getBackendGraphPtr(uint32_t Id) noexcept {
    auto *GraphInst = getGraph(Id);
    if (GraphInst == nullptr || GraphInst->getBackend() != B) {
      return nullptr;
    }
    return &GraphInst->template get<B>();
  }
  template <Backend B>
  detail::BackendContextT<B> *getBackendContextPtr(uint32_t Id) noexcept {
    auto *ContextInst = getContext(Id);
    if (ContextInst == nullptr || ContextInst->getBackend() != B) {
      return nullptr;
    }
    return &ContextInst->template get<B>();
  }
  template <Backend B>
  detail::BackendGraphT<B> *
  getBackendGraphFromContextPtr(uint32_t Id) noexcept {
    auto *ContextInst = getContext(Id);
    if (ContextInst == nullptr || ContextInst->getBackend() != B) {
      return nullptr;
    }
    return getBackendGraphPtr<B>(ContextInst->getGraphId());
  }
  template <Backend B>
  std::tuple<detail::BackendContextT<B> *, detail::BackendGraphT<B> *>
  getBackendContextGraphPtr(uint32_t Id) noexcept {
    auto *ContextInst = getBackendContextPtr<B>(Id);
    auto *GraphInst = getBackendGraphFromContextPtr<B>(Id);
    return {ContextInst, GraphInst};
  }

public:
  template <Backend B> struct BackendState {
    detail::BackendContextT<B> *ContextRef;
    detail::BackendGraphT<B> *GraphRef;

    detail::BackendContextT<B> &context() const noexcept {
      assuming(ContextRef != nullptr);
      return *ContextRef;
    }
    detail::BackendGraphT<B> &graph() const noexcept {
      assuming(GraphRef != nullptr);
      return *GraphRef;
    }
  };
  template <Backend B>
  Expected<detail::BackendGraphT<B> *, ErrNo>
  getBackendGraphOrError(uint32_t Id, std::string_view FunctionName) noexcept {
    using namespace std::literals;
    auto *GraphInst = getBackendGraphPtr<B>(Id);
    if (GraphInst == nullptr) {
      spdlog::error(
          "[WASI-NN] {}: Graph ID {} does not exist or is not a {} graph."sv,
          FunctionName, Id, getBackendName(B));
      return Unexpected<ErrNo>(ErrNo::InvalidArgument);
    }
    return GraphInst;
  }
  template <Backend B>
  Expected<detail::BackendContextT<B> *, ErrNo>
  getBackendContextOrError(uint32_t Id,
                           std::string_view FunctionName) noexcept {
    using namespace std::literals;
    auto *ContextInst = getBackendContextPtr<B>(Id);
    if (ContextInst == nullptr) {
      spdlog::error(
          "[WASI-NN] {}: Context ID {} does not exist or is not a {} context."sv,
          FunctionName, Id, getBackendName(B));
      return Unexpected<ErrNo>(ErrNo::InvalidArgument);
    }
    return ContextInst;
  }
  template <Backend B>
  Expected<BackendState<B>, ErrNo>
  getBackendContextGraphOrError(uint32_t Id,
                                std::string_view FunctionName) noexcept {
    using namespace std::literals;
    auto *ContextInst = getContext(Id);
    if (ContextInst == nullptr || ContextInst->getBackend() != B) {
      spdlog::error(
          "[WASI-NN] {}: Context ID {} does not exist or is not a {} context."sv,
          FunctionName, Id, getBackendName(B));
      return Unexpected<ErrNo>(ErrNo::InvalidArgument);
    }
    auto GraphInst =
        getBackendGraphOrError<B>(ContextInst->getGraphId(), FunctionName);
    if (!GraphInst) {
      return Unexpected<ErrNo>(GraphInst.error());
    }
    return BackendState<B>{&ContextInst->template get<B>(), *GraphInst};
  }
  template <Backend B, typename FuncT>
  Expect<ErrNo> withBackendGraph(uint32_t Id, std::string_view FunctionName,
                                 FuncT &&Func) noexcept {
    auto GraphInst = getBackendGraphOrError<B>(Id, FunctionName);
    if (!GraphInst) {
      return GraphInst.error();
    }
    return std::forward<FuncT>(Func)(**GraphInst);
  }
  template <Backend B, typename FuncT>
  Expect<ErrNo> withBackendContext(uint32_t Id, std::string_view FunctionName,
                                   FuncT &&Func) noexcept {
    auto ContextInst = getBackendContextOrError<B>(Id, FunctionName);
    if (!ContextInst) {
      return ContextInst.error();
    }
    return std::forward<FuncT>(Func)(**ContextInst);
  }
  template <Backend B, typename FuncT>
  Expect<ErrNo> withBackendState(uint32_t Id, std::string_view FunctionName,
                                 FuncT &&Func) noexcept {
    auto State = getBackendContextGraphOrError<B>(Id, FunctionName);
    if (!State) {
      return State.error();
    }
    return std::forward<FuncT>(Func)(State->context(), State->graph());
  }

private:
  template <Backend B>
  detail::BackendGraphT<B> &getBackendGraph(uint32_t Id) noexcept {
    auto *GraphInst = getBackendGraphPtr<B>(Id);
    assuming(GraphInst != nullptr);
    return *GraphInst;
  }
  template <Backend B>
  detail::BackendContextT<B> &getBackendContext(uint32_t Id) noexcept {
    auto *ContextInst = getBackendContextPtr<B>(Id);
    assuming(ContextInst != nullptr);
    return *ContextInst;
  }
  template <Backend B>
  detail::BackendGraphT<B> &getBackendGraphFromContext(uint32_t Id) noexcept {
    auto *GraphInst = getBackendGraphFromContextPtr<B>(Id);
    assuming(GraphInst != nullptr);
    return *GraphInst;
  }
  template <Backend B>
  std::tuple<detail::BackendContextT<B> &, detail::BackendGraphT<B> &>
  getBackendContextGraph(uint32_t Id) noexcept {
    auto [ContextInst, GraphInst] = getBackendContextGraphPtr<B>(Id);
    assuming(ContextInst != nullptr);
    assuming(GraphInst != nullptr);
    return {*ContextInst, *GraphInst};
  }

public:
  void setGraphReady(uint32_t Id) noexcept;
  void setContextGraphReady(uint32_t Id) noexcept;
  void setContextGraphInvalid(uint32_t Id) noexcept;
  void deleteGraph(const uint32_t Id) noexcept;
  void deleteContext(const uint32_t Id) noexcept;
#ifdef WASMEDGE_WASI_NN_TESTING
  void swapGraphContextForTesting(std::deque<Graph> &Graphs,
                                  std::deque<Context> &Contexts) noexcept {
    GraphsStore.swapGraphContextForTesting(Graphs, Contexts);
  }
  void
  registerPreloadModelForTesting(const std::string &Name,
                                 std::vector<std::vector<uint8_t>> Builders,
                                 Backend Encoding, Device Target) noexcept;
#endif

private:
  Graph *getGraph(uint32_t Id) noexcept;
  Context *getContext(uint32_t Id) noexcept;
  Context *getReadyContext(uint32_t Id) noexcept;
  void setGraphInvalid(uint32_t Id) noexcept;

  std::unique_ptr<PreloadStore> Preloads;
  GraphStore GraphsStore;
  mutable std::shared_mutex OperationMutex;

public:
  // Preload model list
  static PO::List<std::string> NNModels;
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  static PO::Option<std::string> NNRPCURI; // For RPC client mode
  bool hasRPCChannel() const noexcept { return NNRPCChannel != nullptr; }
  const std::shared_ptr<grpc::Channel> &getRPCChannel() const noexcept {
    return NNRPCChannel;
  }
#endif

  const Host::WASI::Environ *getEnv() const noexcept { return Environ; }
  void setEnviron(const Runtime::CallingFrame *CurrentFrame) noexcept;

private:
  uint32_t newGraph(Backend BE) noexcept;
  uint32_t newContext(uint32_t GId) noexcept;
  void setContextReady(uint32_t Id) noexcept;

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  std::shared_ptr<grpc::Channel> NNRPCChannel;
#endif
  const Host::WASI::Environ *Environ = nullptr;
};

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
