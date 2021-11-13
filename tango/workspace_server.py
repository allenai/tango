import logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Tuple

from tango.workspace import Workspace


logger = logging.getLogger(__name__)


class WorkspaceRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write("Let's Tango!".encode("UTF-8"))


class WorkspaceServer(ThreadingHTTPServer):
    def __init__(self, address: Tuple[str, int], workspace: Workspace):
        super().__init__(address, WorkspaceRequestHandler)
        self.workspace = workspace

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        logger.info("Server started at %s:%d" % self.server_address)
        super().serve_forever(poll_interval)

    def serve_in_background(self):
        thread = Thread(target=self.serve_forever, name="WebServer", daemon=True)
        thread.start()

    @classmethod
    def on_free_port(cls, workspace: Workspace, start_port: int = 8080) -> "WorkspaceServer":
        for port in range(start_port, 2 ** 16):
            try:
                return cls(("", port), workspace)
            except OSError as e:
                if e.errno == 48:
                    continue
        raise RuntimeError("Could not find free port for server")
