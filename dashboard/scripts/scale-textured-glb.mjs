import fs from "node:fs";
import path from "node:path";

function argument(name) {
  const index = process.argv.indexOf(name);
  const value = index === -1 ? null : process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${name}`);
  return value;
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
  const binary = Buffer.from(file.subarray(binaryHeader + 8, binaryHeader + 8 + binaryLength));
  return { json, binary };
}

function makeGlb(json, binary) {
  const jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  const jsonLength = align4(jsonBytes.length);
  const binaryLength = align4(binary.length);
  const output = Buffer.alloc(12 + 8 + jsonLength + 8 + binaryLength, 0x20);
  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(output.length, 8);
  output.writeUInt32LE(jsonLength, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonBytes.copy(output, 20);
  const binaryHeader = 20 + jsonLength;
  output.writeUInt32LE(binaryLength, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  return output;
}

const input = path.resolve(argument("--input"));
const output = path.resolve(argument("--output"));
const reportPath = process.argv.includes("--report")
  ? path.resolve(argument("--report"))
  : null;
const scale = Number(argument("--scale"));
if (!Number.isFinite(scale) || scale <= 0) throw new Error("--scale must be positive");

const { json, binary } = parseGlb(input);
const positionAccessors = new Set();
for (const mesh of json.meshes ?? []) {
  for (const primitive of mesh.primitives ?? []) {
    if (primitive.attributes?.POSITION != null) {
      positionAccessors.add(primitive.attributes.POSITION);
    }
  }
}
if (!positionAccessors.size) throw new Error("GLB does not contain mesh positions");

let scaledVertices = 0;
for (const accessorIndex of positionAccessors) {
  const accessor = json.accessors[accessorIndex];
  const view = json.bufferViews[accessor.bufferView];
  if (accessor.componentType !== 5126 || accessor.type !== "VEC3") {
    throw new Error("Only float32 VEC3 position accessors are supported");
  }
  if (view.byteStride && view.byteStride !== 12) {
    throw new Error("Interleaved position buffers are not supported");
  }
  const byteOffset = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const positions = new Float32Array(
    binary.buffer,
    binary.byteOffset + byteOffset,
    accessor.count * 3,
  );
  for (let index = 0; index < positions.length; index += 1) positions[index] *= scale;
  accessor.min = accessor.min?.map((value) => value * scale);
  accessor.max = accessor.max?.map((value) => value * scale);
  scaledVertices += accessor.count;
}

json.asset = {
  ...json.asset,
  generator: "ReconBot metric GLB scaler",
  extras: {
    ...(json.asset?.extras ?? {}),
    units: "meters",
    meters_per_input_unit: scale,
  },
};

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, makeGlb(json, binary));
const report = {
  input,
  output,
  units: "meters",
  meters_per_input_unit: scale,
  scaled_vertices: scaledVertices,
};
if (reportPath) {
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
}
console.log(JSON.stringify(report, null, 2));
