#include "HTTPServer.h"


void HttpServerThread::run()
{
    DBG("API server starting");
    svr.Get("/hi", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Hello World!", "text/plain");
    });

    // Run the server in a blocking loop until stopThread() is called
    svr.listen("0.0.0.0", 8080);

}

void HttpServerThread::stopServer()
{
    DBG("API server shutting down");

    svr.stop();
    stopThread(1000); // Gracefully stop thread
}
    
