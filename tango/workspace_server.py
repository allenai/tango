import logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from threading import Thread
from typing import Tuple
from glob import glob
import sys
import json

from tango.workspace import Workspace

logger = logging.getLogger(__name__)

# get json for the ui viz for a list of steps
def run_map(run_name, workspace):
    stepMap = workspace.registered_run(run_name)
    steps = {stepMap.get(sub).unique_id : serialize_step(stepMap.get(sub)) for sub in list(stepMap)}
    return steps

def serialize_step(step):
    return {
        "unique_id": step.unique_id,
        "step_name": step.step_name,
        "step_class_name": step.step_class_name,
        "version": step.version,
        "dependencies": list(step.dependencies),
        "start_time": step.start_time.isoformat(),
        "end_time": step.end_time.isoformat(),
        "error": step.error,
        "result_location": step.result_location,
        "status": step.status
    }

class WorkspaceRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'report/index.html'
        elif self.path == '/api/steps':
            self.send_response(200)
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            output_data = {sub : run_map(sub, self.server.workspace) for sub in list(self.server.workspace.registered_runs())}
            output_json = json.dumps(output_data)
            self.wfile.write(output_json.encode('utf-8'))
            return;
        return SimpleHTTPRequestHandler.do_GET(self)

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
