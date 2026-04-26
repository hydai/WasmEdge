// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinnenv.h"
#include "wasinn_backend.h"
#include "wasinnmodule.h"
#include "wasinntypes.h"

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
#include <grpc/grpc.h>
#endif

using namespace std::literals;

namespace WasmEdge {
namespace Host {

namespace WASINN {

class PreloadStore {
public:
  bool get(const std::string &Name, uint32_t &GraphId) noexcept;
  void removeById(uint32_t GraphId) noexcept;
  void registerModel(const std::string &Name,
                     std::vector<std::vector<uint8_t>> Builders,
                     Backend Encoding, Device Target);
  Expect<WASINN::ErrNo> build(WasiNNEnvironment &Env, const std::string &Name,
                              uint32_t &GraphId,
                              WasiNNEnvironment::Callback Load,
                              std::vector<uint8_t> Config);

private:
  struct PreloadModel {
    std::vector<std::vector<uint8_t>> Builders;
    Backend Encoding;
    Device Target;
  };

  struct BuildState {
    bool Ready = false;
    uint32_t GraphId = 0;
    std::optional<Expect<WASINN::ErrNo>> Result;
    std::condition_variable CV;
  };

  std::mutex Mutex;
  std::unordered_map<std::string, PreloadModel> RawMdMap;
  std::unordered_map<std::string, uint32_t> MdMap;
  std::unordered_map<std::string, std::shared_ptr<BuildState>> PendingMdMap;
};

namespace {
Runtime::Instance::ModuleInstance *
create(const Plugin::PluginModule::ModuleDescriptor *) noexcept {
  return new WasiNNModule;
}

struct DeviceNameMapping {
  std::string_view Name;
  Device Target;
};

constexpr std::array DeviceNameMappings{
    DeviceNameMapping{"cpu"sv, Device::CPU},
    DeviceNameMapping{"gpu"sv, Device::GPU},
    DeviceNameMapping{"tpu"sv, Device::TPU},
    DeviceNameMapping{"auto"sv, Device::AUTO},
};

struct PreloadCommand {
  std::string Name;
  std::string Encoding;
  std::string Target;
  std::vector<std::string> Paths;
};

void toLower(std::string &Value) {
  std::transform(Value.begin(), Value.end(), Value.begin(),
                 [](unsigned char C) {
                   return static_cast<unsigned char>(std::tolower(C));
                 });
}

PreloadCommand parsePreloadCommand(const std::string &Command) {
  std::istringstream ISS(Command);
  const char Delimiter = ':';
  PreloadCommand Result;
  std::getline(ISS, Result.Name, Delimiter);
  std::getline(ISS, Result.Encoding, Delimiter);
  std::getline(ISS, Result.Target, Delimiter);
  std::string Path;
  while (std::getline(ISS, Path, Delimiter)) {
    Result.Paths.push_back(Path);
  }
  toLower(Result.Encoding);
  toLower(Result.Target);
  return Result;
}

const Device *getPreloadDevice(std::string_view Name) noexcept {
  for (const auto &Mapping : DeviceNameMappings) {
    if (Mapping.Name == Name) {
      return &Mapping.Target;
    }
  }
  return nullptr;
}

bool loadPreloadFile(const std::filesystem::path &Path,
                     std::vector<uint8_t> &Data) {
  std::ifstream File(Path, std::ios::binary);
  if (!File.is_open()) {
    spdlog::error("[WASI-NN] Preload model fail."sv);
    return false;
  }
  File.seekg(0, std::ios::end);
  std::streampos FileSize = File.tellg();
  File.seekg(0, std::ios::beg);
  Data.resize(FileSize);
  File.read(reinterpret_cast<char *>(Data.data()), FileSize);
  File.close();
  return true;
}

std::string joinPreloadPaths(const std::vector<std::string> &Paths) {
  std::string Result;
  for (size_t I = 0; I < Paths.size(); ++I) {
    if (I > 0) {
      Result.push_back(':');
    }
    Result += Paths[I];
  }
  return Result;
}

std::vector<uint8_t>
packPathPreloadModel(const std::vector<std::string> &Paths) {
  std::string ModelPath = "preload:";
  ModelPath += joinPreloadPaths(Paths);
  return {ModelPath.begin(), ModelPath.end()};
}

std::vector<std::vector<uint8_t>>
loadPreloadModels(const std::vector<std::string> &Paths,
                  BackendPreloadMode Mode) {
  std::vector<std::vector<uint8_t>> Models;
  Models.reserve(Paths.size());
  if (Mode == BackendPreloadMode::Path) {
    Models.push_back(packPathPreloadModel(Paths));
    return Models;
  }

  for (const std::string &P : Paths) {
    std::vector<uint8_t> Model;
    if (loadPreloadFile(std::filesystem::u8path(P), Model)) {
      Models.push_back(std::move(Model));
    }
  }
  return Models;
}

} // namespace

bool PreloadStore::get(const std::string &Name, uint32_t &GraphId) noexcept {
  std::lock_guard Lock(Mutex);
  if (auto It = MdMap.find(Name); It != MdMap.end()) {
    GraphId = static_cast<uint32_t>(It->second);
    return true;
  }
  return false;
}

void PreloadStore::removeById(uint32_t GraphId) noexcept {
  std::lock_guard Lock(Mutex);
  for (auto It = MdMap.begin(); It != MdMap.end();) {
    if (It->second == static_cast<uint32_t>(GraphId)) {
      It = MdMap.erase(It);
    } else {
      ++It;
    }
  }
}

void PreloadStore::registerModel(const std::string &Name,
                                 std::vector<std::vector<uint8_t>> Builders,
                                 Backend Encoding, Device Target) {
  std::lock_guard Lock(Mutex);
  RawMdMap[Name] = PreloadModel{std::move(Builders), Encoding, Target};
}

Expect<WASINN::ErrNo>
PreloadStore::build(WasiNNEnvironment &Env, const std::string &Name,
                    uint32_t &GraphId, WasiNNEnvironment::Callback Load,
                    std::vector<uint8_t> Config) {
  std::shared_ptr<BuildState> State;
  PreloadModel Model;
  {
    std::unique_lock Lock(Mutex);
    if (auto It = MdMap.find(Name); It != MdMap.end()) {
      GraphId = static_cast<uint32_t>(It->second);
      return WASINN::ErrNo::Success;
    }

    auto RawIt = RawMdMap.find(Name);
    if (RawIt == RawMdMap.end()) {
      return WASINN::ErrNo::NotFound;
    }

    if (auto PendingIt = PendingMdMap.find(Name);
        PendingIt != PendingMdMap.end()) {
      State = PendingIt->second;
      while (!State->Ready) {
        State->CV.wait(Lock);
      }
      assuming(State->Result.has_value());
      if (State->Result->has_value() &&
          State->Result->value() == ErrNo::Success) {
        GraphId = State->GraphId;
      }
      return *State->Result;
    }

    Model = RawIt->second;
    State = std::make_shared<BuildState>();
    PendingMdMap[Name] = State;
  }

  auto &RawMd = Model.Builders;
  std::vector<Span<uint8_t>> Builders;
  Builders.reserve(RawMd.size() + (Config.empty() ? 0U : 1U));
  for (auto &Builder : RawMd) {
    Builders.emplace_back(Builder);
  }
  if (!Config.empty()) {
    Builders.emplace_back(Config);
  }

  auto Result = Load(Env, Builders, Model.Encoding, Model.Target, GraphId);
  {
    std::lock_guard Lock(Mutex);
    if (Result.has_value() && Result.value() == ErrNo::Success &&
        RawMdMap.find(Name) != RawMdMap.end()) {
      MdMap[Name] = GraphId;
      State->GraphId = GraphId;
    }
    State->Result.emplace(Result);
    State->Ready = true;
    PendingMdMap.erase(Name);
  }
  State->CV.notify_all();
  return Result;
}

WasiNNEnvironment::WasiNNEnvironment() noexcept
    : Preloads(std::make_unique<PreloadStore>()) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (getenv("_WASI_NN_RPCSERVER") == nullptr) {
    // RPC client mode
    auto URI = NNRPCURI.value();
    if (!URI.empty()) {
      std::string_view UnixPrefix = "unix://";
      if (URI.substr(0, UnixPrefix.length()) != UnixPrefix) {
        spdlog::warn("[WASI-NN] Expected \"unix://...\", got \"{}\""sv, URI);
      }
      auto Cred = grpc::InsecureChannelCredentials(); // safe for unix://...
      NNRPCChannel = grpc::CreateChannel(URI, Cred);
      if (NNModels.value().size() > 0) {
        spdlog::warn(
            "[WASI-NN] nn-preload has to be specified on the RPC server side, not on the client side"sv);
      }
      return;
    }
  }
#endif
  // Preload NN Models
  for (const auto &M : NNModels.value()) {
    auto Command = parsePreloadCommand(M);
    auto *Selection = getBackendSelection(Command.Encoding);
    auto *TargetDevice = getPreloadDevice(Command.Target);
    if (Selection != nullptr && TargetDevice != nullptr) {
      auto Models = loadPreloadModels(Command.Paths, Selection->Preload);
      Preloads->registerModel(Command.Name, std::move(Models), Selection->Type,
                              *TargetDevice);
    } else {
      spdlog::error(
          "[WASI-NN] Preload Model's Backend or Device is Not Support."sv);
    }
  }
}

WasiNNEnvironment::~WasiNNEnvironment() noexcept = default;

bool WasiNNEnvironment::mdGet(const std::string &Name,
                              uint32_t &GraphId) noexcept {
  return Preloads->get(Name, GraphId);
}

void WasiNNEnvironment::mdRemoveById(uint32_t GraphId) noexcept {
  Preloads->removeById(GraphId);
}

Expect<WASINN::ErrNo>
WasiNNEnvironment::mdBuild(const std::string &Name, uint32_t &GraphId,
                           Callback Load,
                           std::vector<uint8_t> Config) noexcept {
  return Preloads->build(*this, Name, GraphId, std::move(Load),
                         std::move(Config));
}

#ifdef WASMEDGE_WASI_NN_TESTING
void WasiNNEnvironment::registerPreloadModelForTesting(
    const std::string &Name, std::vector<std::vector<uint8_t>> Builders,
    Backend Encoding, Device Target) noexcept {
  Preloads->registerModel(Name, std::move(Builders), Encoding, Target);
}
#endif

void WasiNNEnvironment::setEnviron(
    const Runtime::CallingFrame *CurrentFrame) noexcept {
  auto *WasiModule = CurrentFrame->getWASIModule();
  if (WasiModule != nullptr) {
    Environ =
        dynamic_cast<const WasmEdge::Host::WasiModule *>(WasiModule)->getEnv();
  }
}

PO::List<std::string> WasiNNEnvironment::NNModels(
    PO::Description(
        "Allow preload models from wasinn plugin. Each NN model can be specified as --nn-preload `COMMAND`."sv),
    PO::MetaVar("COMMANDS"sv));

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
PO::Option<std::string> WasiNNEnvironment::NNRPCURI(
    PO::Description("Specify NN RPC URI to connect (\"unix://...\")"sv),
    PO::MetaVar("URI"sv), PO::DefaultValue(std::string("")));
#endif

namespace {
void addOptions(const Plugin::Plugin::PluginDescriptor *,
                PO::ArgumentParser &Parser) noexcept {
  Parser.add_option("nn-preload"sv, WasiNNEnvironment::NNModels);
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (getenv("_WASI_NN_RPCSERVER") == nullptr) {
    // RPC client mode
    Parser.add_option("nn-rpc-uri"sv, WasiNNEnvironment::NNRPCURI);
  }
#endif
}

static Plugin::PluginModule::ModuleDescriptor MD[] = {
    {
        /* Name */ "wasi_nn",
        /* Description */ "",
        /* Create */ create,
    },
};

Plugin::Plugin::PluginDescriptor Descriptor{
    /* Name */ "wasi_nn",
    /* Description */ "",
    /* APIVersion */ Plugin::Plugin::CurrentAPIVersion,
    /* Version */
    {WASI_NN_VERSION_MAJOR, WASI_NN_VERSION_MINOR, WASI_NN_VERSION_PATCH, 0},
    /* ModuleCount */ 1,
    /* ModuleDescriptions */ MD,
    /* ComponentCount */ 0,
    /* ComponentDescriptions */ nullptr,
    /* AddOptions */ addOptions,
};
} // namespace

EXPORT_GET_DESCRIPTOR(Descriptor)

} // namespace WASINN

} // namespace Host
} // namespace WasmEdge
