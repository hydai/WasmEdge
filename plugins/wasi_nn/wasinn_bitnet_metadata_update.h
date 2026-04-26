// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_bitnet.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include <string>

namespace WasmEdge::Host::WASINN::BitNet {

struct MetadataUpdateSnapshot {
  int64_t NGPULayers;
  int64_t MainGpu;
  int64_t Threads;
  bool FlashAttn;
  int64_t CtxSize;
  bool Embedding;
  double Temp;
  double TopP;
  double RepeatPenalty;
  double PresencePenalty;
  double FrequencyPenalty;
  std::string Grammar;
  uint64_t Seed;
};

MetadataUpdateSnapshot
captureMetadataUpdateSnapshot(const Graph &GraphRef) noexcept;

void updateMetadataChangeFlags(const Graph &GraphRef,
                               const MetadataUpdateSnapshot &Snapshot,
                               bool *IsModelUpdated, bool *IsContextUpdated,
                               bool *IsSamplerUpdated) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
