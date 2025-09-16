#include "httplib.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <etcd/Client.hpp>
#include <string>

class MasterHttpServer {
  public:
    explicit MasterHttpServer(const std::string &etcdEndpoints, int httpPort);
    ~MasterHttpServer();
    void run();

  private:
    etcd::Client etcdClient_;
    httplib::Server httpServer_;
    int httpPort_;
    std::set<std::string> key_set;

    void setResponse(httplib::Response &res, int retCode,
                     const std::string &msg,
                     const rapidjson::Document *data = nullptr);
    void addIndexNode(const httplib::Request &req,
                     httplib::Response &res);
    void addStorageNode(const httplib::Request &req,
                       httplib::Response &res);
    void getIndexNode(const httplib::Request &req,
                     httplib::Response &res);
    void getStorageNode(const httplib::Request &req,
                       httplib::Response &res);
};