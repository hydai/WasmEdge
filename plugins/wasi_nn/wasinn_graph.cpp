// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinnenv.h"

#include <deque>
#include <mutex>
#include <shared_mutex>

namespace WasmEdge {
namespace Host {
namespace WASINN {
namespace {

bool isRecycled(const std::unordered_set<uint32_t> &Recycle,
                uint32_t Id) noexcept {
  return Recycle.find(Id) != Recycle.end();
}

template <typename T>
bool isActiveSlot(const T &Slots, const std::unordered_set<uint32_t> &Recycle,
                  uint32_t Id) noexcept {
  return Id < Slots.size() && !isRecycled(Recycle, Id);
}

template <typename T>
void releaseSlot(std::deque<T> &Slots, std::unordered_set<uint32_t> &Recycle,
                 uint32_t Id) noexcept {
  if (Id == Slots.size() - 1) {
    Slots.pop_back();
  } else {
    Slots[Id].reset();
    Recycle.insert(Id);
  }
}

} // namespace

uint32_t GraphStore::newGraph(Backend BE) noexcept {
  std::unique_lock Lock(GraphMutex);
  uint32_t ID = static_cast<uint32_t>(NNGraph.size());
  if (NNGraphRecycle.empty()) {
    NNGraph.emplace_back(BE);
  } else {
    ID = *NNGraphRecycle.begin();
    NNGraph[ID].init(BE);
    NNGraphRecycle.erase(ID);
  }
  return ID;
}

uint32_t GraphStore::newContext(uint32_t GId) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (!isActiveSlot(NNGraph, NNGraphRecycle, GId) || !NNGraph[GId].isReady()) {
    return InvalidId;
  }
  auto &G = NNGraph[GId];
  uint32_t ID = static_cast<uint32_t>(NNContext.size());
  if (NNContextRecycle.empty()) {
    NNContext.emplace_back(GId, G);
  } else {
    ID = *NNContextRecycle.begin();
    NNContext[ID].init(GId, G);
    NNContextRecycle.erase(ID);
  }
  G.increaseContext();
  return ID;
}

bool GraphStore::hasGraph(uint32_t Id) noexcept {
  return getGraph(Id) != nullptr;
}

Graph *GraphStore::getGraph(uint32_t Id) noexcept {
  std::shared_lock Lock(GraphMutex);
  if (isActiveSlot(NNGraph, NNGraphRecycle, Id)) {
    return &NNGraph[Id];
  }
  return nullptr;
}

bool GraphStore::isGraphReady(uint32_t Id) noexcept {
  auto *G = getGraph(Id);
  return G != nullptr && G->isReady();
}

bool GraphStore::isGraphFinalized(uint32_t Id) noexcept {
  auto *G = getGraph(Id);
  return G != nullptr && G->isFinalized();
}

std::optional<Backend> GraphStore::getGraphBackend(uint32_t Id) noexcept {
  auto *G = getGraph(Id);
  if (G == nullptr) {
    return std::nullopt;
  }
  return G->getBackend();
}

Context *GraphStore::getContext(uint32_t Id) noexcept {
  std::shared_lock Lock(GraphMutex);
  if (isActiveSlot(NNContext, NNContextRecycle, Id)) {
    return &NNContext[Id];
  }
  return nullptr;
}

bool GraphStore::hasContext(uint32_t Id) noexcept {
  return getContext(Id) != nullptr;
}

Context *GraphStore::getReadyContext(uint32_t Id) noexcept {
  auto *C = getContext(Id);
  if (C != nullptr && C->isReady()) {
    return C;
  }
  return nullptr;
}

bool GraphStore::isContextReady(uint32_t Id) noexcept {
  return getReadyContext(Id) != nullptr;
}

std::optional<Backend> GraphStore::getContextBackend(uint32_t Id) noexcept {
  auto *C = getContext(Id);
  if (C == nullptr) {
    return std::nullopt;
  }
  return C->getBackend();
}

bool GraphStore::isContextGraphReady(uint32_t Id) noexcept {
  auto *C = getContext(Id);
  return C != nullptr && isGraphReady(C->getGraphId());
}

std::optional<uint32_t> GraphStore::getContextGraphId(uint32_t Id) noexcept {
  auto *C = getContext(Id);
  if (C == nullptr) {
    return std::nullopt;
  }
  return C->getGraphId();
}

void GraphStore::setGraphReady(uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (isActiveSlot(NNGraph, NNGraphRecycle, Id)) {
    NNGraph[Id].setReady();
  }
}

void GraphStore::setGraphInvalid(uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (isActiveSlot(NNGraph, NNGraphRecycle, Id)) {
    NNGraph[Id].setInvalid();
  }
}

void GraphStore::setContextGraphReady(uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (!isActiveSlot(NNContext, NNContextRecycle, Id)) {
    return;
  }
  const auto GId = NNContext[Id].getGraphId();
  if (isActiveSlot(NNGraph, NNGraphRecycle, GId)) {
    NNGraph[GId].setReady();
  }
}

void GraphStore::setContextGraphInvalid(uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (!isActiveSlot(NNContext, NNContextRecycle, Id)) {
    return;
  }
  const auto GId = NNContext[Id].getGraphId();
  if (isActiveSlot(NNGraph, NNGraphRecycle, GId)) {
    NNGraph[GId].setInvalid();
  }
}

void GraphStore::setContextReady(uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (isActiveSlot(NNContext, NNContextRecycle, Id)) {
    NNContext[Id].setReady();
  }
}

void GraphStore::deleteGraph(const uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (!isActiveSlot(NNGraph, NNGraphRecycle, Id)) {
    return;
  }
  auto &G = NNGraph[Id];
  G.setFinalized();
  if (G.getContextCount() == 0) {
    releaseSlot(NNGraph, NNGraphRecycle, Id);
  }
}

void GraphStore::deleteContext(const uint32_t Id) noexcept {
  std::unique_lock Lock(GraphMutex);
  if (!isActiveSlot(NNContext, NNContextRecycle, Id)) {
    return;
  }
  auto GId = NNContext[Id].getGraphId();
  if (isActiveSlot(NNGraph, NNGraphRecycle, GId)) {
    auto &G = NNGraph[GId];
    G.decreaseContext();
    if (G.getContextCount() == 0 && G.isFinalized()) {
      releaseSlot(NNGraph, NNGraphRecycle, GId);
    }
  }
  releaseSlot(NNContext, NNContextRecycle, Id);
}

uint32_t WasiNNEnvironment::newGraph(Backend BE) noexcept {
  return GraphsStore.newGraph(BE);
}

uint32_t WasiNNEnvironment::newContext(uint32_t GId) noexcept {
  return GraphsStore.newContext(GId);
}

bool WasiNNEnvironment::hasGraph(uint32_t Id) noexcept {
  return GraphsStore.hasGraph(Id);
}

Graph *WasiNNEnvironment::getGraph(uint32_t Id) noexcept {
  return GraphsStore.getGraph(Id);
}

bool WasiNNEnvironment::isGraphReady(uint32_t Id) noexcept {
  return GraphsStore.isGraphReady(Id);
}

bool WasiNNEnvironment::isGraphFinalized(uint32_t Id) noexcept {
  return GraphsStore.isGraphFinalized(Id);
}

std::optional<Backend>
WasiNNEnvironment::getGraphBackend(uint32_t Id) noexcept {
  return GraphsStore.getGraphBackend(Id);
}

Context *WasiNNEnvironment::getContext(uint32_t Id) noexcept {
  return GraphsStore.getContext(Id);
}

bool WasiNNEnvironment::hasContext(uint32_t Id) noexcept {
  return GraphsStore.hasContext(Id);
}

Context *WasiNNEnvironment::getReadyContext(uint32_t Id) noexcept {
  return GraphsStore.getReadyContext(Id);
}

bool WasiNNEnvironment::isContextReady(uint32_t Id) noexcept {
  return GraphsStore.isContextReady(Id);
}

std::optional<Backend>
WasiNNEnvironment::getContextBackend(uint32_t Id) noexcept {
  return GraphsStore.getContextBackend(Id);
}

bool WasiNNEnvironment::isContextGraphReady(uint32_t Id) noexcept {
  return GraphsStore.isContextGraphReady(Id);
}

std::optional<uint32_t>
WasiNNEnvironment::getContextGraphId(uint32_t Id) noexcept {
  return GraphsStore.getContextGraphId(Id);
}

void WasiNNEnvironment::setGraphReady(uint32_t Id) noexcept {
  GraphsStore.setGraphReady(Id);
}

void WasiNNEnvironment::setGraphInvalid(uint32_t Id) noexcept {
  GraphsStore.setGraphInvalid(Id);
}

void WasiNNEnvironment::setContextGraphReady(uint32_t Id) noexcept {
  GraphsStore.setContextGraphReady(Id);
}

void WasiNNEnvironment::setContextGraphInvalid(uint32_t Id) noexcept {
  GraphsStore.setContextGraphInvalid(Id);
}

void WasiNNEnvironment::setContextReady(uint32_t Id) noexcept {
  GraphsStore.setContextReady(Id);
}

void WasiNNEnvironment::deleteGraph(const uint32_t Id) noexcept {
  GraphsStore.deleteGraph(Id);
}

void WasiNNEnvironment::deleteContext(const uint32_t Id) noexcept {
  GraphsStore.deleteContext(Id);
}

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
