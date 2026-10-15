#!/usr/bin/env node
/**
 * Run Python with the project's interpreter.
 *
 * The backend has dependencies (fastapi, langgraph, psycopg, ortools) that the
 * system python does not have. Every npm script that touches the backend goes
 * through here so `npm run build` and `npm test` work whether or not the user
 * has activated a virtualenv.
 *
 * Resolution order:
 *   1. $PYTHON, if set
 *   2. backend/.venv/bin/python
 *   3. .venv/bin/python
 *   4. python3
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const candidates = [
  process.env.PYTHON,
  path.join(root, "backend", ".venv", "bin", "python"),
  path.join(root, ".venv", "bin", "python"),
  "python3",
].filter(Boolean);

const interpreter = candidates.find((c) => c === "python3" || existsSync(c));

if (!interpreter) {
  console.error("Could not find a Python interpreter. Set $PYTHON.");
  process.exit(1);
}

if (interpreter !== process.env.PYTHON && !existsSync(interpreter)) {
  console.error(`Python interpreter not found: ${interpreter}`);
  process.exit(1);
}

const result = spawnSync(interpreter, process.argv.slice(2), {
  stdio: "inherit",
  cwd: root,
});

process.exit(result.status ?? 1);
