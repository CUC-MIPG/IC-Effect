import http.server
import socketserver
from pathlib import Path


def run(port: int = 8000) -> None:
    """Serve the current directory so you can preview the homepage locally."""
    root = Path(__file__).parent.resolve()
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {root} at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            httpd.server_close()


if __name__ == "__main__":
    run()
