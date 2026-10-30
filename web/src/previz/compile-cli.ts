// stdin: one sentence. stdout: the compiled spec, what matched, and what did not.
import { compile, type CameraPick } from "./compile.ts";

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

const picked = process.env.PREVIZ_CAMERA;
const camera = picked === "close" || picked === "wide" || picked === "close-wide" || picked === "dolly-in" || picked === "dolly-out" || picked === "orbit" || picked === "drone" || picked === "fpv" || picked === "track" || picked === "parallel" || picked === "locked" ? picked as CameraPick : null;
const out = compile(await readStdin(), { camera });
process.stdout.write(JSON.stringify({ spec: out.spec, title: out.title, matched: out.matched, missed: out.missed, camera: out.camera }));
