import fs from "node:fs";
import path from "node:path";

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${name}`);
  return value;
}

function numberList(name) {
  const index = process.argv.indexOf(name);
  if (index === -1) return null;
  const values = process.argv.slice(index + 1, index + 4).map(Number);
  if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} requires three numeric values`);
  }
  return values;
}

function align4(value) {
  return (value + 3) & ~3;
}

function parseGlb(filename) {
  const file = fs.readFileSync(filename);
  if (file.readUInt32LE(0) !== 0x46546c67 || file.readUInt32LE(4) !== 2) {
    throw new Error("Input must be a GLB 2.0 file");
  }
  const jsonLength = file.readUInt32LE(12);
  const json = JSON.parse(
    file.subarray(20, 20 + jsonLength).toString("utf8").replace(/[\s\0]+$/, ""),
  );
  const binaryHeader = 20 + jsonLength;
  const binaryLength = file.readUInt32LE(binaryHeader);
  const binary = file.subarray(binaryHeader + 8, binaryHeader + 8 + binaryLength);
  return { json, binary };
}

function accessorArray(json, binary, accessorIndex, ArrayType, width) {
  const accessor = json.accessors[accessorIndex];
  const view = json.bufferViews[accessor.bufferView];
  const byteOffset = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  return new ArrayType(
    binary.buffer,
    binary.byteOffset + byteOffset,
    accessor.count * width,
  );
}

function makeGlb(json, binary) {
  const jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  const jsonLength = align4(jsonBytes.length);
  const binaryLength = align4(binary.length);
  const totalLength = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = Buffer.alloc(totalLength, 0x20);
  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(jsonLength, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonBytes.copy(output, 20);
  const binaryHeader = 20 + jsonLength;
  output.writeUInt32LE(binaryLength, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  return output;
}

function positionKey(positions, index) {
  const offset = index * 3;
  return `${positions[offset]},${positions[offset + 1]},${positions[offset + 2]}`;
}

const input = path.resolve(argument("--input"));
const output = path.resolve(argument("--output"));
const reportPath = argument("--report");
const min = numberList("--min");
const max = numberList("--max");
const keepLargest = process.argv.includes("--largest-component");
if ((min && !max) || (!min && max)) throw new Error("Use --min and --max together");

const { json, binary } = parseGlb(input);
const primitive = json.meshes?.[0]?.primitives?.[0];
if (!primitive) throw new Error("GLB does not contain the expected mesh primitive");

const positions = accessorArray(
  json,
  binary,
  primitive.attributes.POSITION,
  Float32Array,
  3,
);
const normals = accessorArray(
  json,
  binary,
  primitive.attributes.NORMAL,
  Float32Array,
  3,
);
const uvs = accessorArray(
  json,
  binary,
  primitive.attributes.TEXCOORD_0,
  Float32Array,
  2,
);
const indices = accessorArray(json, binary, primitive.indices, Uint32Array, 1);
const faceCount = indices.length / 3;
const candidateFaces = [];

for (let face = 0; face < faceCount; face += 1) {
  let inside = true;
  if (min && max) {
    for (let corner = 0; corner < 3 && inside; corner += 1) {
      const vertex = indices[face * 3 + corner] * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const value = positions[vertex + axis];
        if (value < min[axis] || value > max[axis]) {
          inside = false;
          break;
        }
      }
    }
  }
  if (inside) candidateFaces.push(face);
}
if (candidateFaces.length === 0) throw new Error("Cleanup bounds removed every face");

let retainedFaces = candidateFaces;
let componentCount = 1;
if (keepLargest) {
  const parent = new Int32Array(candidateFaces.length);
  const sizes = new Int32Array(candidateFaces.length);
  const firstFaceByPosition = new Map();
  for (let index = 0; index < candidateFaces.length; index += 1) {
    parent[index] = index;
    sizes[index] = 1;
  }
  const find = (value) => {
    let current = value;
    while (parent[current] !== current) {
      parent[current] = parent[parent[current]];
      current = parent[current];
    }
    return current;
  };
  const union = (left, right) => {
    let a = find(left);
    let b = find(right);
    if (a === b) return;
    if (sizes[a] < sizes[b]) [a, b] = [b, a];
    parent[b] = a;
    sizes[a] += sizes[b];
  };

  for (let localFace = 0; localFace < candidateFaces.length; localFace += 1) {
    const sourceFace = candidateFaces[localFace];
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = indices[sourceFace * 3 + corner];
      const key = positionKey(positions, vertex);
      const prior = firstFaceByPosition.get(key);
      if (prior == null) firstFaceByPosition.set(key, localFace);
      else union(localFace, prior);
    }
  }

  const counts = new Map();
  for (let index = 0; index < candidateFaces.length; index += 1) {
    const root = find(index);
    counts.set(root, (counts.get(root) ?? 0) + 1);
  }
  componentCount = counts.size;
  const largestRoot = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  retainedFaces = candidateFaces.filter((_, index) => find(index) === largestRoot);
}

const cleanedPositions = new Float32Array(retainedFaces.length * 9);
const cleanedNormals = new Float32Array(retainedFaces.length * 9);
const cleanedUvs = new Float32Array(retainedFaces.length * 6);
const cleanedIndices = new Uint32Array(retainedFaces.length * 3);
const retainedPositionKeys = new Set();
const outputMin = [Infinity, Infinity, Infinity];
const outputMax = [-Infinity, -Infinity, -Infinity];

for (let targetFace = 0; targetFace < retainedFaces.length; targetFace += 1) {
  const sourceFace = retainedFaces[targetFace];
  for (let corner = 0; corner < 3; corner += 1) {
    const sourceVertex = indices[sourceFace * 3 + corner];
    retainedPositionKeys.add(positionKey(positions, sourceVertex));
    const targetVertex = targetFace * 3 + corner;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[sourceVertex * 3 + axis];
      cleanedPositions[targetVertex * 3 + axis] = value;
      cleanedNormals[targetVertex * 3 + axis] = normals[sourceVertex * 3 + axis];
      outputMin[axis] = Math.min(outputMin[axis], value);
      outputMax[axis] = Math.max(outputMax[axis], value);
    }
    cleanedUvs[targetVertex * 2] = uvs[sourceVertex * 2];
    cleanedUvs[targetVertex * 2 + 1] = uvs[sourceVertex * 2 + 1];
    cleanedIndices[targetVertex] = targetVertex;
  }
}

const imageView = json.bufferViews[json.images[0].bufferView];
const image = binary.subarray(
  imageView.byteOffset ?? 0,
  (imageView.byteOffset ?? 0) + imageView.byteLength,
);
const arrays = [cleanedPositions, cleanedNormals, cleanedUvs, cleanedIndices];
const chunks = [];
const views = [];
let offset = 0;
for (const array of arrays) {
  const aligned = align4(offset);
  if (aligned > offset) chunks.push(Buffer.alloc(aligned - offset));
  offset = aligned;
  const buffer = Buffer.from(array.buffer, array.byteOffset, array.byteLength);
  views.push({ byteOffset: offset, byteLength: buffer.length });
  chunks.push(buffer);
  offset += buffer.length;
}
const imageOffset = align4(offset);
if (imageOffset > offset) chunks.push(Buffer.alloc(imageOffset - offset));
chunks.push(image);
const cleanedBinary = Buffer.concat(chunks);

const cleanedJson = {
  ...json,
  asset: {
    ...json.asset,
    generator: "ReconBot textured GLB cleaner",
    extras: {
      source_faces: faceCount,
      retained_faces: retainedFaces.length,
      crop_min: min,
      crop_max: max,
      largest_component: keepLargest,
    },
  },
  buffers: [{ byteLength: cleanedBinary.length }],
  bufferViews: [
    { buffer: 0, ...views[0], target: 34962 },
    { buffer: 0, ...views[1], target: 34962 },
    { buffer: 0, ...views[2], target: 34962 },
    { buffer: 0, ...views[3], target: 34963 },
    { buffer: 0, byteOffset: imageOffset, byteLength: image.length },
  ],
  accessors: [
    {
      bufferView: 0,
      componentType: 5126,
      count: cleanedPositions.length / 3,
      type: "VEC3",
      min: outputMin,
      max: outputMax,
    },
    {
      bufferView: 1,
      componentType: 5126,
      count: cleanedNormals.length / 3,
      type: "VEC3",
    },
    {
      bufferView: 2,
      componentType: 5126,
      count: cleanedUvs.length / 2,
      type: "VEC2",
    },
    {
      bufferView: 3,
      componentType: 5125,
      count: cleanedIndices.length,
      type: "SCALAR",
      min: [0],
      max: [cleanedIndices.length - 1],
    },
  ],
};
cleanedJson.images = [{ ...json.images[0], bufferView: 4 }];
cleanedJson.meshes[0].primitives[0].attributes = {
  POSITION: 0,
  NORMAL: 1,
  TEXCOORD_0: 2,
};
cleanedJson.meshes[0].primitives[0].indices = 3;

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, makeGlb(cleanedJson, cleanedBinary));
const report = {
  input,
  output,
  source_faces: faceCount,
  source_vertices: positions.length / 3,
  crop_candidate_faces: candidateFaces.length,
  retained_faces: retainedFaces.length,
  retained_unique_vertices: retainedPositionKeys.size,
  removed_faces: faceCount - retainedFaces.length,
  components_after_crop: componentCount,
  bounds: { min: outputMin, max: outputMax },
};
if (reportPath) {
  const resolvedReport = path.resolve(reportPath);
  fs.mkdirSync(path.dirname(resolvedReport), { recursive: true });
  fs.writeFileSync(resolvedReport, `${JSON.stringify(report, null, 2)}\n`);
}
console.log(JSON.stringify(report, null, 2));
