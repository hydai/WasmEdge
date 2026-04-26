// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_rpc_client.h"

#ifdef WASMEDGE_BUILD_WASI_NN_RPC

#include "wasi_ephemeral_nn.grpc.pb.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#include "common/spdlog.h"

#include <charconv>
#include <map>
#include <string_view>
#include <system_error>
#include <utility>

#include <grpc/grpc.h>

namespace WasmEdge {
namespace Host {
namespace WASINN {
namespace {
using namespace std::literals;

ErrNo metadataToErrNo(
    const std::multimap<grpc::string_ref, grpc::string_ref> &Metadata) {
  const auto ErrNoIt = Metadata.find("errno");
  if (ErrNoIt != Metadata.end()) {
    const auto &Value = ErrNoIt->second;
    if (Value.length() == 0) {
      return ErrNo::RuntimeError;
    }

    const auto *Begin = Value.data();
    const auto *End = Begin + Value.length();
    uint32_t RawErrNo = 0;
    const auto [Parsed, Error] = std::from_chars(Begin, End, RawErrNo);
    if (Error != std::errc{} || Parsed != End) {
      return ErrNo::RuntimeError;
    }
    return static_cast<ErrNo>(RawErrNo);
  }
  return ErrNo::Success;
}

template <typename Call> ErrNo callRpc(Call &&CallFn) {
  grpc::ClientContext ClientContext;
  const auto Status = std::forward<Call>(CallFn)(ClientContext);
  if (!Status.ok()) {
    return metadataToErrNo(ClientContext.GetServerTrailingMetadata());
  }
  return ErrNo::Success;
}

template <typename ServiceT, typename RequestT, typename ResponseT,
          typename MethodT>
ErrNo callUnary(WasiNNEnvironment &Env, RequestT &Req, ResponseT &Res,
                MethodT Method) {
  auto Stub = ServiceT::NewStub(Env.getRPCChannel());
  return callRpc([&](grpc::ClientContext &ClientContext) {
    return (Stub.get()->*Method)(&ClientContext, Req, &Res);
  });
}

template <typename RequestT>
RequestT resourceRequest(uint32_t ResourceHandle) noexcept {
  RequestT Req;
  Req.set_resource_handle(ResourceHandle);
  return Req;
}

ErrNo copyRpcOutputData(std::string_view Data, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  return copyStringToBuffer(Data, OutBuffer, BytesWritten);
}

Expect<ErrNo> unsupportedRpcOperation(std::string_view Operation) noexcept {
  spdlog::error("[WASI-NN] RPC client is not implemented for {}"sv, Operation);
  return ErrNo::UnsupportedOperation;
}

} // namespace

namespace RPC {

Expect<ErrNo> load() noexcept { return unsupportedRpcOperation("load"sv); }

Expect<ErrNo> loadByName(WasiNNEnvironment &Env, Span<const uint8_t> Name,
                         uint32_t &GraphId) {
  wasi_ephemeral_nn::LoadByNameRequest Req;
  auto NameStrView = asStringView(Name);
  Req.set_name(NameStrView.data(), NameStrView.size());
  wasi_ephemeral_nn::LoadByNameResult Res;
  auto Err = callUnary<wasi_ephemeral_nn::Graph>(
      Env, Req, Res, &wasi_ephemeral_nn::Graph::Stub::LoadByName);
  if (Err != ErrNo::Success) {
    return Err;
  }
  GraphId = Res.graph_handle();
  return ErrNo::Success;
}

Expect<ErrNo> loadByNameWithConfig(WasiNNEnvironment &Env,
                                   Span<const uint8_t> Name,
                                   Span<const uint8_t> Config,
                                   uint32_t &GraphId) {
  wasi_ephemeral_nn::LoadByNameWithConfigRequest Req;
  auto NameStrView = asStringView(Name);
  auto ConfigStrView = asStringView(Config);
  Req.set_name(NameStrView.data(), NameStrView.size());
  Req.set_config(ConfigStrView.data(), ConfigStrView.size());
  wasi_ephemeral_nn::LoadByNameWithConfigResult Res;
  auto Err = callUnary<wasi_ephemeral_nn::Graph>(
      Env, Req, Res, &wasi_ephemeral_nn::Graph::Stub::LoadByNameWithConfig);
  if (Err != ErrNo::Success) {
    return Err;
  }
  GraphId = Res.graph_handle();
  return ErrNo::Success;
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) {
  auto Req =
      resourceRequest<wasi_ephemeral_nn::InitExecutionContextRequest>(GraphId);
  wasi_ephemeral_nn::InitExecutionContextResult Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphResource::Stub::InitExecutionContext);
  if (Err != ErrNo::Success) {
    return Err;
  }
  ContextId = Res.ctx_handle();
  return ErrNo::Success;
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env,
                       Runtime::Instance::MemoryInstance &MemInst,
                       uint32_t ContextId, uint32_t Index,
                       const DecodedTensor &Tensor) {
  auto Req = resourceRequest<wasi_ephemeral_nn::SetInputRequest>(ContextId);
  Req.set_index(Index);
  wasi_ephemeral_nn::Tensor RPCTensor;
  RPCTensor.mutable_dimensions()->Add(Tensor.Data.Dimension.begin(),
                                      Tensor.Data.Dimension.end());
  RPCTensor.set_ty(wasi_ephemeral_nn::TensorType(Tensor.Data.RType));
  RPCTensor.set_data(MemInst.getPointer<char *>(Tensor.RawTensorPtr),
                     Tensor.RawTensorLen);
  *Req.mutable_tensor() = RPCTensor;
  google::protobuf::Empty Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::SetInput);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return ErrNo::Success;
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) {
  auto Req = resourceRequest<wasi_ephemeral_nn::GetOutputRequest>(ContextId);
  Req.set_index(Index);
  wasi_ephemeral_nn::GetOutputResult Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::GetOutput);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return copyRpcOutputData(Res.data(), OutBuffer, BytesWritten);
}

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten) {
  auto Req = resourceRequest<wasi_ephemeral_nn::GetOutputRequest>(ContextId);
  Req.set_index(Index);
  wasi_ephemeral_nn::GetOutputResult Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::GetOutputSingle);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return copyRpcOutputData(Res.data(), OutBuffer, BytesWritten);
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) {
  auto Req = resourceRequest<wasi_ephemeral_nn::ComputeRequest>(ContextId);
  google::protobuf::Empty Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::Compute);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return ErrNo::Success;
}

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env, uint32_t ContextId) {
  auto Req = resourceRequest<wasi_ephemeral_nn::ComputeRequest>(ContextId);
  google::protobuf::Empty Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::ComputeSingle);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return ErrNo::Success;
}

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId) {
  auto Req = resourceRequest<wasi_ephemeral_nn::FiniSingleRequest>(ContextId);
  google::protobuf::Empty Res;
  auto Err = callUnary<wasi_ephemeral_nn::GraphExecutionContextResource>(
      Env, Req, Res,
      &wasi_ephemeral_nn::GraphExecutionContextResource::Stub::FiniSingle);
  if (Err != ErrNo::Success) {
    return Err;
  }
  return ErrNo::Success;
}

Expect<ErrNo> unload() noexcept { return unsupportedRpcOperation("unload"sv); }

Expect<ErrNo> finalizeExecCtx() noexcept {
  return unsupportedRpcOperation("finalize_execution_context"sv);
}

} // namespace RPC

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge

#endif
