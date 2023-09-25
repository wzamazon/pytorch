#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <c10/util/CallOnce.h>

#ifdef USE_C10D_NCCL

#include <mutex>
#include <cassert>
#include <netdb.h>
#include <sys/socket.h>
#include <arpa/inet.h>

namespace c10d {

void** getGlobalConnData()
{
    static void** globalConnData = NULL;

    if (globalConnData == NULL) {
        const char *worldSizeStr = getenv("REAL_WORLD_SIZE");
        if (worldSizeStr == NULL)
            throw std::runtime_error("Error: REAL_WORLD_SIZE was not set");

        const char *nodeCountStr = getenv("WORLD_SIZE");
        if (nodeCountStr == NULL)
            throw std::runtime_error("Error: WORLD_SIZE was not set");

        const char *jobName = getenv("JOB_NAME");
        if (jobName == NULL)
            throw std::runtime_error("Error: JOB_NAME was not set");

        int worldSize = atoi(worldSizeStr);
        assert(worldSize > 0);
        int nodeCount = atoi(nodeCountStr);
        assert(nodeCount > 0);

        int procPerNode = worldSize / nodeCount;
        if (worldSize != procPerNode * nodeCount)
            throw std::runtime_error("Error: invalid REAL_WORLD_SIZE and WORLD_SIZE");
        assert(procPerNode > 0);
        globalConnData = (void**)malloc(worldSize * sizeof(void*));
        for (int i = 0; i < nodeCount; ++i) {
            std::ostringstream nodeNameOss;
            if (i == 0) {
                nodeNameOss << jobName << "-master-0";
            } else {
                nodeNameOss << jobName << "-worker-" << (i - 1);
            }

	    std::string nodeName = nodeNameOss.str();
	    struct addrinfo *addrInfo;
            if (getaddrinfo(nodeName.c_str(), NULL, NULL, &addrInfo)!=0) {
                // TODO: fix the memory leak of globalConnData here
                throw std::runtime_error(("Error: getaddrinfo for " + nodeName + " failed!").c_str());
            }

            int portBase = 976; // use priviliaged port to avoid collision
            for (int j = 0; j < procPerNode; ++j) {
                int globalRank = i * procPerNode + j;
                globalConnData[globalRank] = (struct sockaddr*)calloc(1, sizeof(struct sockaddr));
                memcpy(globalConnData[globalRank], addrInfo->ai_addr, sizeof(struct sockaddr));

                int port = portBase + j;
                struct sockaddr* sock_addr = (struct sockaddr*)globalConnData[globalRank];

                if (sock_addr->sa_family == AF_INET) {
                    struct sockaddr_in *sin = (struct sockaddr_in*)sock_addr;
                    sin->sin_port = htons(port);
                } else if (sock_addr->sa_family == AF_INET6) {
                    struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)sock_addr;
                    sin6->sin6_port = htons(port);
                } else {
                    throw std::runtime_error("Error: unknown AF family");
                }
            }
        }
    }

    assert(globalConnData);
    return globalConnData;
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
