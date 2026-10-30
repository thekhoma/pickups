import { readFileSync } from "node:fs";
import { frameZeroDigest } from "./cameraMath.ts";

const spec = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(frameZeroDigest(spec));
