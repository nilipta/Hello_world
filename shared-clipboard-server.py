import http.server
import socketserver
import json

PORT = 8000
shared_clipboard = []

class ClipboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the shared clipboard HTML file as the default root page
        if self.path == '/':
            self.path = '/shared-clipboard.html'
            
        if self.path == '/api/clipboard':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps({'content': shared_clipboard}).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        global shared_clipboard
        if self.path == '/api/clipboard':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                new_content = data.get('content', '')
                if new_content:
                    shared_clipboard.append(new_content)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    with http.server.HTTPServer(("", PORT), ClipboardHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()