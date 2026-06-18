import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dashboardRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.join(dashboardRoot, "dist");
const libraryRoot = path.resolve(
  process.env.RECONBOT_LIBRARY_ROOT ||
    path.join(dashboardRoot, "public", "reconstructions"),
);
const catalogPath = path.join(libraryRoot, "catalog.json");
const port = Number(process.env.PORT || 4173);
const host = process.env.HOST || "127.0.0.1";
const mime = {
  ".css": "text/css; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function readCatalog() {
  if (!fs.existsSync(catalogPath)) {
    return { active_id: null, items: [] };
  }
  return JSON.parse(fs.readFileSync(catalogPath, "utf8"));
}

function writeCatalog(catalog) {
  fs.mkdirSync(libraryRoot, { recursive: true });
  const temporary = `${catalogPath}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, catalogPath);
}

function catalogItems(catalog) {
  return catalog.items ?? catalog.records ?? [];
}

function replaceCatalogItems(catalog, items) {
  if (Array.isArray(catalog.records)) catalog.records = items;
  else catalog.items = items;
}

function publicCatalog() {
  const catalog = readCatalog();
  return {
    active_id: catalog.active_id,
    items: catalogItems(catalog).map((item) => ({
      ...item,
      model_url: `/api/reconstructions/${encodeURIComponent(item.id)}/model.glb`,
      video_url: item.asset_files?.video
        ? `/api/reconstructions/${encodeURIComponent(item.id)}/video.mp4`
        : null,
    })),
  };
}

function json(response, status, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  response.end(body);
}

function safeIdentifier(value) {
  return /^[A-Za-z0-9_-]{1,80}$/.test(value);
}

function recordFor(identifier) {
  const catalog = readCatalog();
  const record = catalogItems(catalog).find((item) => item.id === identifier);
  return { catalog, record };
}

function streamFile(request, response, file) {
  const stat = fs.statSync(file);
  const extension = path.extname(file).toLowerCase();
  const range = request.headers.range;
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!match) {
      response.writeHead(416, { "Content-Range": `bytes */${stat.size}` }).end();
      return;
    }
    const start = match[1] ? Number(match[1]) : 0;
    const end = match[2] ? Math.min(Number(match[2]), stat.size - 1) : stat.size - 1;
    if (start > end || start >= stat.size) {
      response.writeHead(416, { "Content-Range": `bytes */${stat.size}` }).end();
      return;
    }
    response.writeHead(206, {
      "Content-Type": mime[extension] || "application/octet-stream",
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-cache",
    });
    fs.createReadStream(file, { start, end }).pipe(response);
    return;
  }
  response.writeHead(200, {
    "Content-Type": mime[extension] || "application/octet-stream",
    "Content-Length": stat.size,
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-cache",
  });
  fs.createReadStream(file).pipe(response);
}

function handleApi(request, response, pathname) {
  if (request.method === "GET" && pathname === "/api/health") {
    json(response, 200, { ok: true, role: "local-preview-library" });
    return true;
  }
  if (request.method === "GET" && pathname === "/api/reconstructions") {
    json(response, 200, publicCatalog());
    return true;
  }

  const match = /^\/api\/reconstructions\/([^/]+)(?:\/(activate|model\.glb|video\.mp4))?$/.exec(
    pathname,
  );
  if (!match) return false;
  const identifier = decodeURIComponent(match[1]);
  const action = match[2];
  if (!safeIdentifier(identifier)) {
    json(response, 400, { error: "Invalid reconstruction ID" });
    return true;
  }
  const { catalog, record } = recordFor(identifier);
  if (!record) {
    json(response, 404, { error: "Reconstruction not found" });
    return true;
  }

  if (request.method === "POST" && action === "activate") {
    catalog.active_id = identifier;
    writeCatalog(catalog);
    json(response, 200, { ok: true, ...publicCatalog() });
    return true;
  }

  if (request.method === "DELETE" && !action) {
    const target = path.resolve(libraryRoot, identifier);
    if (path.dirname(target) !== libraryRoot) {
      json(response, 403, { error: "Invalid reconstruction path" });
      return true;
    }
    fs.rmSync(target, { recursive: true, force: true });
    replaceCatalogItems(
      catalog,
      catalogItems(catalog).filter((item) => item.id !== identifier),
    );
    if (catalog.active_id === identifier) {
      const remaining = [...catalogItems(catalog)].sort((a, b) =>
        String(b.created_at || "").localeCompare(String(a.created_at || "")),
      );
      catalog.active_id = remaining[0]?.id ?? null;
    }
    writeCatalog(catalog);
    json(response, 200, { ok: true, ...publicCatalog() });
    return true;
  }

  if (request.method === "GET" && (action === "model.glb" || action === "video.mp4")) {
    const kind = action === "model.glb" ? "model" : "video";
    const filename = record.asset_files?.[kind];
    const file = filename
      ? path.resolve(libraryRoot, identifier, path.basename(filename))
      : null;
    if (
      !file ||
      path.dirname(file) !== path.join(libraryRoot, identifier) ||
      !fs.existsSync(file)
    ) {
      json(response, 404, { error: "Reconstruction asset not found" });
      return true;
    }
    streamFile(request, response, file);
    return true;
  }

  json(response, 405, { error: "Method not allowed" });
  return true;
}

http
  .createServer((request, response) => {
    try {
      const pathname = decodeURIComponent((request.url || "/").split("?")[0]);
      if (handleApi(request, response, pathname)) return;

      const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
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
    } catch (error) {
      json(response, 500, {
        error: error instanceof Error ? error.message : "Preview server error",
      });
    }
  })
  .listen(port, host, () => {
    console.log(`ReconBot dashboard: http://${host}:${port}`);
    console.log(`Reconstruction library: ${libraryRoot}`);
  });
