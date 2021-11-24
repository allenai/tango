import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Dict, Tuple

import click

from tango.common.logging import click_logger
from tango.workspace import StepInfo, Workspace


class WorkspaceRequestHandler(SimpleHTTPRequestHandler):
    @classmethod
    def _run_map(cls, run_name: str, workspace: Workspace):
        step_map = workspace.registered_run(run_name)
        # TODO: This needs to return all dependencies as well.
        return {
            step_info.unique_id: cls._serialize_step_info(step_info)
            for step_info in step_map.values()
        }

    @classmethod
    def _serialize_step_info(cls, step_info: StepInfo) -> Dict[str, Any]:
        return {
            "unique_id": step_info.unique_id,
            "step_name": step_info.step_name,
            "step_class_name": step_info.step_class_name,
            "version": step_info.version,
            "dependencies": list(step_info.dependencies),
            "start_time": step_info.start_time.isoformat() if step_info.start_time else None,
            "end_time": step_info.end_time.isoformat() if step_info.end_time else None,
            "error": step_info.error,
            "result_location": step_info.result_location,
            "state": str(step_info.state),
        }

    def do_GET(self):
        if self.path == "/":
            self.path = "report/index.html"
        if self.path.startswith("/run/"):
            self.path = "report/report.html"
        elif self.path == "/api/stepinfo":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            output_data = {
                sub: self._run_map(sub, self.server.workspace)
                for sub in list(self.server.workspace.registered_runs())
            }
            output_json = json.dumps(output_data)
            self.wfile.write(output_json.encode("utf-8"))
            return
        elif self.path == "/api/runlist":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            output_data = list(self.server.workspace.registered_runs())
            output_json = json.dumps(output_data)
            self.wfile.write(output_json.encode("utf-8"))
            return
        return SimpleHTTPRequestHandler.do_GET(self)


class WorkspaceServer(ThreadingHTTPServer):
    def __init__(self, address: Tuple[str, int], workspace: Workspace):
        super().__init__(address, WorkspaceRequestHandler)
        self.workspace = workspace

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        address, port = self.server_address
        if address == "0.0.0.0":
            address = "localhost"  # Safari has a problem with 0.0.0.0
        click_logger.info("Server started at " + click.style(f"http://{address}:{port}", bold=True))
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
