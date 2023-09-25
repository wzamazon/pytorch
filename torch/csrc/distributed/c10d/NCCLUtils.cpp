#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <c10/util/CallOnce.h>

#ifdef USE_C10D_NCCL

#include <mutex>
#include <cassert>
#include <iostream>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

namespace c10d {

struct GlobalConnData& getGlobalConnData()
{
    static struct GlobalConnData globalConnData;

    if (globalConnData.addresses.size() == 0) {
        const char *worldSizeStr = getenv("REAL_WORLD_SIZE");
        if (worldSizeStr == NULL)
            throw std::runtime_error("Error: REAL_WORLD_SIZE was not set");

        const char *nodeCountStr = getenv("NUM_NODES");
        if (nodeCountStr == NULL)
            throw std::runtime_error("Error: NUM_NODES was not set");

        const char *jobName = getenv("JOB_NAME");
        if (jobName == NULL)
            throw std::runtime_error("Error: JOB_NAME was not set");

        globalConnData.worldSize = atoi(worldSizeStr);
        assert(globalConnData.worldSize > 0);
        globalConnData.nodeCount = atoi(nodeCountStr);
        assert(globalConnData.nodeCount > 0);

        globalConnData.procPerNode = globalConnData.worldSize / globalConnData.nodeCount;
        if (globalConnData.worldSize != globalConnData.procPerNode * globalConnData.nodeCount)
            throw std::runtime_error("Error: invalid REAL_WORLD_SIZE and WORLD_SIZE");
        assert(globalConnData.procPerNode > 0);

	std::cerr << getpid() <<  ": pytorch nodeCount=" << globalConnData.nodeCount << " worldSize=" << globalConnData.worldSize << " procPerNode=" << globalConnData.procPerNode << std::endl;
        globalConnData.addresses.resize(globalConnData.worldSize);
        for (int i = 0; i < globalConnData.nodeCount; ++i) {
            std::ostringstream nodeNameOss;
            if (i == 0) {
                nodeNameOss << jobName << "-master-0";
            } else {
                nodeNameOss << jobName << "-worker-" << (i - 1);
            }

	    std::string nodeName = nodeNameOss.str();
	    struct addrinfo *addrInfo;
            if (getaddrinfo(nodeName.c_str(), NULL, NULL, &addrInfo)!=0) {
                throw std::runtime_error(("Error: getaddrinfo for " + nodeName + " failed!").c_str());
            }

            for (int j = 0; j < globalConnData.procPerNode; ++j) {
                int globalRank = i * globalConnData.procPerNode + j;
                memcpy(&globalConnData.addresses[globalRank], addrInfo->ai_addr, sizeof(struct sockaddr));
            }

            freeaddrinfo(addrInfo);
        }
    }

    assert(globalConnData.addresses);
    return globalConnData;
}

void setConnDataPort(void *connData, int port) {
    struct sockaddr* sockAddr = (struct sockaddr *)connData;

    if (sockAddr->sa_family == AF_INET) {
        struct sockaddr_in *sin = (struct sockaddr_in*)sockAddr;
        sin->sin_port = htons(port);
    } else if (sockAddr->sa_family == AF_INET6) {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)sockAddr;
        sin6->sin6_port = htons(port);
    } else {
        throw std::runtime_error("Error: unknown AF family");
    }
}

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK(
        false,
        c10::str(
            "NCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  return ncclComm_;
}

std::string getNcclVersion() {
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      // NCCL changed version coding starting 2.9
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      auto ncclMajor = version / majorBase;
      auto ncclMinor = (version % majorBase) / minorBase;
      auto ncclPatch =
          version % (ncclMajor * majorBase + ncclMinor * minorBase);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
std::string getNcclErrorDetailStr(
    ncclResult_t error,
    c10::optional<std::string> processGroupFailureReason /* = c10::nullopt */
) {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != c10::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  err = "\nLast error:\n" + std::string(ncclGetLastError(NULL));
#endif
  switch (error) {
    case ncclUnhandledCudaError:
      interpret = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      interpret =
          "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // Before ncclRemoteError was created, unexpected remote disconnect was
      // categorized as ncclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
#endif
      break;
    case ncclInternalError:
      interpret = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      interpret = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      interpret =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      interpret =
          "ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
#endif
    default:
      interpret = "Unknown NCCL error!";
  }
  return interpret + err;
}

} // namespace c10d

#endif // USE_C10D_NCCL
