import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..", "..");

function optionValue(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${name}`);
  }
  return path.resolve(root, value);
}

const sourcePly = optionValue(
  "--ply",
  path.join(
    root,
    "outputs",
    "openmvs",
    "session_003",
    "metric",
    "cropped",
    "scene_dense_texture_300k_metric_cropped.ply",
  ),
);
const sourceTexture = optionValue(
  "--texture",
  path.join(
    root,
    "outputs",
    "openmvs",
    "session_003",
    "metric",
    "cropped",
    "scene_dense_texture_300k0.png",
  ),
);
const outputDir = optionValue(
  "--out-dir",
  path.join(root, "dashboard", "public", "assets"),
);
const outputGlb = optionValue(
  "--out-glb",
  path.join(outputDir, "reconstruction.glb"),
);
const outputTexture = optionValue(
  "--out-texture",
  path.join(outputDir, "reconstruction-texture.png"),
);

function align4(value) {
  return (value + 3) & ~3;
}

function makeGlb(json, binary) {
  const jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  const paddedJsonLength = align4(jsonBytes.length);
  const paddedBinaryLength = align4(binary.length);
  const totalLength = 12 + 8 + paddedJsonLength + 8 + paddedBinaryLength;
  const output = Buffer.alloc(totalLength, 0x20);

  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(paddedJsonLength, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonBytes.copy(output, 20);

  const binaryHeader = 20 + paddedJsonLength;
  output.writeUInt32LE(paddedBinaryLength, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  return output;
}

const ply = fs.readFileSync(sourcePly);
const headerEndMarker = Buffer.from("end_header\n");
const headerEnd = ply.indexOf(headerEndMarker) + headerEndMarker.length;
if (headerEnd < headerEndMarker.length) {
  throw new Error("Invalid PLY header");
}
const header = ply.subarray(0, headerEnd).toString("utf8");
const vertexCount = Number(header.match(/element vertex (\d+)/)?.[1]);
const faceCount = Number(header.match(/element face (\d+)/)?.[1]);
if (!vertexCount || !faceCount) {
  throw new Error("PLY vertex or face count missing");
}

const vertexBytes = vertexCount * 12;
const positionsSource = new Float32Array(vertexCount * 3);
for (let i = 0; i < positionsSource.length; i += 1) {
  positionsSource[i] = ply.readFloatLE(headerEnd + i * 4);
}
const faceOffset = headerEnd + vertexBytes;
const faceStride = 1 + 12 + 1 + 24;

// OpenMVS stores UVs per face corner. Duplicate vertices to preserve texture seams.
const positions = new Float32Array(faceCount * 9);
const normals = new Float32Array(faceCount * 9);
const uvs = new Float32Array(faceCount * 6);
const indices = new Uint32Array(faceCount * 3);

for (let face = 0; face < faceCount; face += 1) {
  const offset = faceOffset + face * faceStride;
  if (ply.readUInt8(offset) !== 3 || ply.readUInt8(offset + 13) !== 6) {
    throw new Error(`Unsupported face record at ${face}`);
  }
  const ids = [
    ply.readUInt32LE(offset + 1),
    ply.readUInt32LE(offset + 5),
    ply.readUInt32LE(offset + 9),
  ];
  const base = face * 9;
  for (let corner = 0; corner < 3; corner += 1) {
    const source = ids[corner] * 3;
    positions[base + corner * 3] = positionsSource[source];
    positions[base + corner * 3 + 1] = positionsSource[source + 1];
    positions[base + corner * 3 + 2] = positionsSource[source + 2];
    uvs[face * 6 + corner * 2] = ply.readFloatLE(offset + 14 + corner * 8);
    uvs[face * 6 + corner * 2 + 1] = 1 - ply.readFloatLE(offset + 18 + corner * 8);
    indices[face * 3 + corner] = face * 3 + corner;
  }

  const ax = positions[base];
  const ay = positions[base + 1];
  const az = positions[base + 2];
  const bx = positions[base + 3];
  const by = positions[base + 4];
  const bz = positions[base + 5];
  const cx = positions[base + 6];
  const cy = positions[base + 7];
  const cz = positions[base + 8];
  const ux = bx - ax;
  const uy = by - ay;
  const uz = bz - az;
  const vx = cx - ax;
  const vy = cy - ay;
  const vz = cz - az;
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  const length = Math.hypot(nx, ny, nz) || 1;
  nx /= length;
  ny /= length;
  nz /= length;
  for (let corner = 0; corner < 3; corner += 1) {
    normals[base + corner * 3] = nx;
    normals[base + corner * 3 + 1] = ny;
    normals[base + corner * 3 + 2] = nz;
  }
}

const texture = fs.readFileSync(sourceTexture);
const chunks = [];
const views = [];
let byteOffset = 0;
for (const array of [positions, normals, uvs, indices]) {
  const buffer = Buffer.from(array.buffer, array.byteOffset, array.byteLength);
  const aligned = align4(byteOffset);
  if (aligned > byteOffset) chunks.push(Buffer.alloc(aligned - byteOffset));
  byteOffset = aligned;
  views.push({ byteOffset, byteLength: buffer.length });
  chunks.push(buffer);
  byteOffset += buffer.length;
}
const imageOffset = align4(byteOffset);
if (imageOffset > byteOffset) chunks.push(Buffer.alloc(imageOffset - byteOffset));
chunks.push(texture);
const binary = Buffer.concat(chunks);

const min = [Infinity, Infinity, Infinity];
const max = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < positions.length; i += 3) {
  for (let axis = 0; axis < 3; axis += 1) {
    min[axis] = Math.min(min[axis], positions[i + axis]);
    max[axis] = Math.max(max[axis], positions[i + axis]);
  }
}

const gltf = {
  asset: { version: "2.0", generator: "ReconBot textured PLY converter" },
  scene: 0,
  scenes: [{ nodes: [0] }],
  nodes: [{ mesh: 0, name: "Metric reconstruction" }],
  meshes: [
    {
      primitives: [
        {
          attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
          indices: 3,
          material: 0,
        },
      ],
    },
  ],
  materials: [
    {
      name: "OpenMVS texture",
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
        metallicFactor: 0,
        roughnessFactor: 0.92,
      },
      doubleSided: true,
    },
  ],
  textures: [{ sampler: 0, source: 0 }],
  samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 }],
  images: [{ bufferView: 4, mimeType: "image/png" }],
  buffers: [{ byteLength: binary.length }],
  bufferViews: [
    { buffer: 0, ...views[0], target: 34962 },
    { buffer: 0, ...views[1], target: 34962 },
    { buffer: 0, ...views[2], target: 34962 },
    { buffer: 0, ...views[3], target: 34963 },
    { buffer: 0, byteOffset: imageOffset, byteLength: texture.length },
  ],
  accessors: [
    {
      bufferView: 0,
      componentType: 5126,
      count: positions.length / 3,
      type: "VEC3",
      min,
      max,
    },
    {
      bufferView: 1,
      componentType: 5126,
      count: normals.length / 3,
      type: "VEC3",
    },
    {
      bufferView: 2,
      componentType: 5126,
      count: uvs.length / 2,
      type: "VEC2",
    },
    {
      bufferView: 3,
      componentType: 5125,
      count: indices.length,
      type: "SCALAR",
      min: [0],
      max: [indices.length - 1],
    },
  ],
};

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(outputGlb, makeGlb(gltf, binary));
fs.copyFileSync(sourceTexture, outputTexture);
console.log(`Wrote ${outputGlb}`);
console.log(`Copied ${outputTexture}`);
