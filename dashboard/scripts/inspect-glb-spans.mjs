import fs from "node:fs";
import path from "node:path";

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${name}`);
  return value;
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

function quantile(values, fraction) {
  if (!values.length) return null;
  const index = Math.max(0, Math.min(values.length - 1, Math.round((values.length - 1) * fraction)));
  return values[index];
}

function numericArgument(name, fallback = null) {
  const value = argument(name, fallback == null ? null : String(fallback));
  if (value == null) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric`);
  return parsed;
}

const input = path.resolve(argument("--input"));
const bins = Number(argument("--bins", "20"));
const requestedYMin = numericArgument("--y-min");
const requestedYMax = numericArgument("--y-max");
const requestedZMin = numericArgument("--z-min");
const requestedZMax = numericArgument("--z-max");
const { json, binary } = parseGlb(input);
const primitive = json.meshes?.[0]?.primitives?.[0];
if (!primitive) throw new Error("GLB does not contain the expected mesh primitive");
const positions = accessorArray(json, binary, primitive.attributes.POSITION, Float32Array, 3);
const accessor = json.accessors[primitive.attributes.POSITION];
const yMin = accessor.min[1];
const yMax = accessor.max[1];

if (requestedYMin != null || requestedYMax != null) {
  const x = [];
  const z = [];
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    const y = positions[vertex + 1];
    const depth = positions[vertex + 2];
    if (requestedYMin != null && y < requestedYMin) continue;
    if (requestedYMax != null && y > requestedYMax) continue;
    if (requestedZMin != null && depth < requestedZMin) continue;
    if (requestedZMax != null && depth > requestedZMax) continue;
    x.push(positions[vertex]);
    z.push(depth);
  }
  x.sort((a, b) => a - b);
  z.sort((a, b) => a - b);
  const centerX = x.reduce((sum, value) => sum + value, 0) / x.length;
  const centerZ = z.reduce((sum, value) => sum + value, 0) / z.length;
  let xx = 0;
  let xz = 0;
  let zz = 0;
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    const y = positions[vertex + 1];
    const depth = positions[vertex + 2];
    if (requestedYMin != null && y < requestedYMin) continue;
    if (requestedYMax != null && y > requestedYMax) continue;
    if (requestedZMin != null && depth < requestedZMin) continue;
    if (requestedZMax != null && depth > requestedZMax) continue;
    const dx = positions[vertex] - centerX;
    const dz = depth - centerZ;
    xx += dx * dx;
    xz += dx * dz;
    zz += dz * dz;
  }
  const angle = 0.5 * Math.atan2(2 * xz, xx - zz);
  const axisX = Math.cos(angle);
  const axisZ = Math.sin(angle);
  const projected = [];
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    const y = positions[vertex + 1];
    const depth = positions[vertex + 2];
    if (requestedYMin != null && y < requestedYMin) continue;
    if (requestedYMax != null && y > requestedYMax) continue;
    if (requestedZMin != null && depth < requestedZMin) continue;
    if (requestedZMax != null && depth > requestedZMax) continue;
    projected.push((positions[vertex] - centerX) * axisX + (depth - centerZ) * axisZ);
  }
  projected.sort((a, b) => a - b);
  const fractions = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995, 0.999, 1];
  const values = Object.fromEntries(
    fractions.map((fraction) => [`x${String(fraction * 100).replace(".", "_")}`, quantile(x, fraction)]),
  );
  console.log(JSON.stringify({
    vertices: x.length,
    filters: {
      y_min: requestedYMin,
      y_max: requestedYMax,
      z_min: requestedZMin,
      z_max: requestedZMax,
    },
    ...values,
    principal_axis_xz: [axisX, axisZ],
    principal_span_robust: quantile(projected, 0.999) - quantile(projected, 0.001),
    principal_span_exact: projected.at(-1) - projected[0],
    z_min: z[0] ?? null,
    z_max: z.at(-1) ?? null,
  }, null, 2));
  process.exit(0);
}

const binHeight = (yMax - yMin) / bins;
const rows = [];

for (let bin = 0; bin < bins; bin += 1) {
  const lower = yMin + bin * binHeight;
  const upper = lower + binHeight;
  const x = [];
  const z = [];
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    const y = positions[vertex + 1];
    if (y < lower || (bin === bins - 1 ? y > upper : y >= upper)) continue;
    x.push(positions[vertex]);
    z.push(positions[vertex + 2]);
  }
  x.sort((a, b) => a - b);
  z.sort((a, b) => a - b);
  const x05 = quantile(x, 0.05);
  const x95 = quantile(x, 0.95);
  const z05 = quantile(z, 0.05);
  const z95 = quantile(z, 0.95);
  rows.push({
    y: `${lower.toFixed(3)}..${upper.toFixed(3)}`,
    vertices: x.length,
    x05: x05?.toFixed(3),
    x95: x95?.toFixed(3),
    x90_span: x05 == null || x95 == null ? null : (x95 - x05).toFixed(3),
    z05: z05?.toFixed(3),
    z95: z95?.toFixed(3),
  });
}

console.table(rows);
