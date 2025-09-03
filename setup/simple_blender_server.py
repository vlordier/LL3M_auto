"""Simple Blender HTTP server for LL3M integration using built-in modules."""

import json
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from urllib.parse import urlparse

import bpy


class BlenderHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Blender operations."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/health":
            self.send_health_response()
        elif path == "/scene/info":
            self.send_scene_info()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
        else:
            data = {}

        if path == "/execute":
            self.execute_code(data)
        elif path == "/scene/save":
            self.save_scene(data)
        else:
            self.send_error(404, "Endpoint not found")

    def send_health_response(self):
        """Send health check response."""
        response = {"status": "healthy", "blender_version": bpy.app.version_string}
        self.send_json_response(response)

    def send_scene_info(self):
        """Send scene information."""
        try:
            scene = bpy.context.scene
            scene_info = {
                "name": scene.name if scene else "Unknown",
                "frame_start": scene.frame_start if scene else 1,
                "frame_end": scene.frame_end if scene else 250,
                "frame_current": scene.frame_current if scene else 1,
                "objects": [obj.name for obj in scene.objects] if scene else [],
            }

            # Try to get active object, but handle headless mode gracefully
            try:
                active_obj = bpy.context.active_object
                scene_info["active_object"] = active_obj.name if active_obj else None
            except AttributeError:
                scene_info["active_object"] = None

            self.send_json_response(scene_info)
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_json_response(error_response, status_code=500)

    def execute_code(self, data):
        """Execute Python code in Blender."""
        code = data.get("code", "")
        if not code:
            self.send_json_response(
                {"success": False, "error": "No code provided"}, 400
            )
            return

        try:
            # Capture output
            stdout_buffer = StringIO()
            stderr_buffer = StringIO()

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Create safe execution environment
                safe_globals = {
                    "bpy": bpy,
                    "print": print,
                    "__builtins__": __builtins__,
                }

                exec(code, safe_globals)

            # Get captured output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()

            logs = []
            if stdout_content:
                logs.append(f"STDOUT: {stdout_content}")
            if stderr_content:
                logs.append(f"STDERR: {stderr_content}")

            response = {"success": True, "result": stdout_content, "logs": logs}
            self.send_json_response(response)

        except Exception as e:
            error_msg = str(e)
            response = {
                "success": False,
                "error": error_msg,
                "logs": [f"ERROR: {error_msg}"],
            }
            self.send_json_response(response)

    def save_scene(self, data):
        """Save Blender scene."""
        filepath = data.get("filepath") or data
        if isinstance(data, str):
            filepath = data

        if not filepath:
            self.send_json_response(
                {"success": False, "error": "No filepath provided"}, 400
            )
            return

        try:
            bpy.ops.wm.save_as_mainfile(filepath=filepath)
            response = {"success": True, "filepath": filepath}
            self.send_json_response(response)
        except Exception as e:
            response = {"success": False, "error": str(e)}
            self.send_json_response(response, 500)

    def send_json_response(self, data, status_code=200):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        json_data = json.dumps(data).encode("utf-8")
        self.wfile.write(json_data)

    def log_message(self, format, *args):
        """Override to reduce log noise."""
        print(f"[{self.address_string()}] {format % args}")


class BlenderServer:
    """Simple HTTP server for Blender."""

    def __init__(self, port=3001):
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        """Start the server."""
        print(f"🚀 Starting Blender HTTP Server on port {self.port}")
        print(f"Access at: http://localhost:{self.port}")

        self.server = HTTPServer(("localhost", self.port), BlenderHandler)

        # Start server in a separate thread so Blender doesn't block
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        print("✅ Blender MCP Server is running")
        print("Endpoints:")
        print("  GET  /health     - Health check")
        print("  GET  /scene/info - Scene information")
        print("  POST /execute    - Execute Python code")
        print("  POST /scene/save - Save scene")
        print("\nPress Ctrl+C to stop")

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            self.stop()

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
        print("✅ Server stopped")


def main():
    """Main entry point."""
    port = 3001

    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 3001")

    # Create and start server
    server = BlenderServer(port)
    server.start()


if __name__ == "__main__":
    main()
