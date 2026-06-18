import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist");
const port = Number(process.env.PORT || 4173);
const host = process.env.HOST || "127.0.0.1";
const mime = {
  ".css": "text/css; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

http
  .createServer((request, response) => {
    const requested = decodeURIComponent((request.url || "/").split("?")[0]);
    const relative = requested === "/" ? "index.html" : requested.replace(/^\/+/, "");
    let file = path.resolve(root, relative);
    if (!file.startsWith(root)) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      file = path.join(root, "index.html");
    }
    const extension = path.extname(file).toLowerCase();
    const cacheControl =
      extension === ".html" || extension === ".glb" || extension === ".png"
        ? "no-cache"
        : "public, max-age=3600";
    response.writeHead(200, {
      "Content-Type": mime[extension] || "application/octet-stream",
      "Cache-Control": cacheControl,
    });
    fs.createReadStream(file).pipe(response);
  })
  .listen(port, host, () => {
    console.log(`ReconBot dashboard: http://${host}:${port}`);
  });
