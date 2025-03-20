#include "httplib.h"
#include <juce_core/juce_core.h>

class HttpServerThread : public juce::Thread {
public:
    HttpServerThread() : juce::Thread("HTTP Server Thread") {}

    void run() override;

    void stopServer();
private:
    httplib::Server svr;
};
