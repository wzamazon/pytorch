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

NCCLConnData::NCCLConnData()
{
    memset(&sockAddr_, 0, sizeof(sockAddr_));
    hostCntWithSameIp_ = 0;
    hostIdxWithSameIp_ = 0;
}

NCCLConnData::NCCLConnData(std::string ipaddr, int hostCntWithSameIp, int hostIdxWithSameIp)
{
    if (ipaddr.find(".") != std::string::npos) {
        sockAddr_.sa_family = AF_INET;
        struct sockaddr_in *sin = (struct sockaddr_in*)&sockAddr_;
        if (inet_pton(AF_INET, ipaddr.c_str(), &sin->sin_addr) != 1) {
            throw std::runtime_error("Error: invalid ipv4 addr" + ipaddr);
        }
    } else if (ipaddr.find(":") != std::string::npos) {
        sockAddr_.sa_family = AF_INET6;
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6*)&sockAddr_;
        if (inet_pton(AF_INET6, ipaddr.c_str(), &sin6->sin6_addr) != 1) {
            throw std::runtime_error("Error: invalid ipv6 addr" + ipaddr);
        }
    } else {
        throw std::runtime_error("Error: unknown ip address family for addr: " + ipaddr);
    }

    hostCntWithSameIp_ = hostCntWithSameIp;
    hostIdxWithSameIp_ = hostIdxWithSameIp;
}

void NCCLConnData::setPort(int port) {
    if (sockAddr_.sa_family == AF_INET) {
        struct sockaddr_in *sin = (struct sockaddr_in*)&sockAddr_;
        sin->sin_port = htons(port);
    } else if (sockAddr_.sa_family == AF_INET6) {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)&sockAddr_;
        sin6->sin6_port = htons(port);
    } else {
        throw std::runtime_error("Error: unknown AF family");
    }
}

std::string NCCLConnData::getIpaddr() const {
    if (sockAddr_.sa_family == AF_INET) {
        char ipaddr[INET_ADDRSTRLEN];
        struct sockaddr_in *sin = (struct sockaddr_in*)&sockAddr_;
        if (inet_ntop(AF_INET, &sin->sin_addr, ipaddr, INET_ADDRSTRLEN) == NULL) {
            throw std::runtime_error("Error: cannot convert ipv4 addr to string");
        }

	return std::string(ipaddr);
    }
   
    if (sockAddr_.sa_family == AF_INET6) {
        char ipaddr[INET6_ADDRSTRLEN];
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6*)&sockAddr_;
        if (inet_ntop(AF_INET6, &sin6->sin6_addr, ipaddr, INET6_ADDRSTRLEN) == NULL) {
            throw std::runtime_error("Error: connot covert ipv6 addr to string");
        }
    }

    throw std::runtime_error("Error: unknown ip address family for addr");
}

int NCCLConnData::getPort() const {
    if (sockAddr_.sa_family == AF_INET) {
        struct sockaddr_in *sin = (struct sockaddr_in*)&sockAddr_;
        return ntohs(sin->sin_port);
    }
   
    if (sockAddr_.sa_family == AF_INET6) {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)&sockAddr_;
	return ntohs(sin6->sin6_port);
    }
   
    throw std::runtime_error("Error: unknown AF family");
}

NCCLConnData NCCLConnData::copy() const {
    NCCLConnData copy;
    memcpy(&copy.sockAddr_, &sockAddr_, sizeof(this->sockAddr_));
    copy.hostCntWithSameIp_ = hostCntWithSameIp_;
    copy.hostIdxWithSameIp_ = hostIdxWithSameIp_;
    return copy;
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
