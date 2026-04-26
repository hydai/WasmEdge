// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_bitnet_metadata_update.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

namespace WasmEdge::Host::WASINN::BitNet {

MetadataUpdateSnapshot
captureMetadataUpdateSnapshot(const Graph &GraphRef) noexcept {
  return MetadataUpdateSnapshot{
      GraphRef.Params.n_gpu_layers,
      GraphRef.Params.main_gpu,
      GraphRef.Params.cpuparams.n_threads,
      GraphRef.Params.flash_attn,
      GraphRef.Params.n_ctx,
      GraphRef.Params.embedding,
      GraphRef.Params.sparams.temp,
      GraphRef.Params.sparams.top_p,
      GraphRef.Params.sparams.penalty_repeat,
      GraphRef.Params.sparams.penalty_present,
      GraphRef.Params.sparams.penalty_freq,
      GraphRef.Params.sparams.grammar,
      GraphRef.Params.sparams.seed,
  };
}

void updateMetadataChangeFlags(const Graph &GraphRef,
                               const MetadataUpdateSnapshot &Snapshot,
                               bool *IsModelUpdated, bool *IsContextUpdated,
                               bool *IsSamplerUpdated) noexcept {
  if (IsModelUpdated && (Snapshot.NGPULayers != GraphRef.Params.n_gpu_layers ||
                         Snapshot.MainGpu != GraphRef.Params.main_gpu)) {
    *IsModelUpdated = true;
  }

  if (IsContextUpdated &&
      (Snapshot.CtxSize != GraphRef.Params.n_ctx ||
       Snapshot.Threads != GraphRef.Params.cpuparams.n_threads ||
       Snapshot.FlashAttn != GraphRef.Params.flash_attn ||
       Snapshot.Embedding != GraphRef.Params.embedding)) {
    *IsContextUpdated = true;
  }

  if (IsSamplerUpdated &&
      (Snapshot.Temp != GraphRef.Params.sparams.temp ||
       Snapshot.TopP != GraphRef.Params.sparams.top_p ||
       Snapshot.RepeatPenalty != GraphRef.Params.sparams.penalty_repeat ||
       Snapshot.PresencePenalty != GraphRef.Params.sparams.penalty_present ||
       Snapshot.FrequencyPenalty != GraphRef.Params.sparams.penalty_freq ||
       Snapshot.Grammar != GraphRef.Params.sparams.grammar ||
       Snapshot.Seed != GraphRef.Params.sparams.seed)) {
    *IsSamplerUpdated = true;
  }
}

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
