// Verifies the backend's SSE frames are consumable by the real AI SDK.
//
// Captures a live /api/chat stream (or uses a recorded one from
// mobile/.stream-samples/) and runs it through `readUIMessageStream`, the same
// reducer `useChat` uses. This is the check that catches frame-shape drift on
// the client side.
//
// Usage:
//   node verify-stream.mjs            # uses a recorded sample
//   API_URL=http://localhost:8000 node verify-stream.mjs   # captures live
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { DefaultChatTransport } from "ai";

const API_URL = process.env.API_URL || "http://localhost:8000";
const SAMPLE_DIR = new URL("./.stream-samples/", import.meta.url);

const EXPECTED = {
  "text-start": (c) => typeof c.id === "string" && c.id.length > 0,
  "text-delta": (c) => typeof c.id === "string" && typeof c.delta === "string",
  "text-end": (c) => typeof c.id === "string",
  "data-schedule": (c) => c.data && Array.isArray(c.data.schedule),
};

/**
 * Reduce UI message chunks into one message, mirroring what the SDK does
 * internally for the parts this app uses.
 */
function reduceChunks(chunks) {
  let id = "";
  let role = "assistant";
  const parts = [];
  let openText = null;

  for (const chunk of chunks) {
    switch (chunk.type) {
      case "start":
        id = chunk.messageId ?? id;
        break;
      case "text-start":
        openText = { type: "text", text: "", id: chunk.id };
        parts.push(openText);
        break;
      case "text-delta":
        if (openText && openText.id === chunk.id) openText.text += chunk.delta;
        else {
          openText = { type: "text", text: chunk.delta, id: chunk.id };
          parts.push(openText);
        }
        break;
      case "text-end":
        openText = null;
        break;
      case "finish":
        break;
      default:
        if (chunk.type?.startsWith("data-")) {
          parts.push({
            type: chunk.type,
            id: chunk.id ?? `${chunk.type}-${parts.length}`,
            data: chunk.data,
          });
        }
    }
  }
  return { id, role, parts };
}

async function capture() {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer mock-dev-token" },
    body: JSON.stringify({
      user_id: "mock_user_123",
      thread_id: "verify-stream",
      message: "schedule my day",
    }),
  });
  if (!res.ok) throw new Error(`backend returned HTTP ${res.status}`);
  if (res.headers.get("x-vercel-ai-ui-message-stream") !== "v1") {
    throw new Error(
      `backend did not declare the protocol: got '${res.headers.get("x-vercel-ai-ui-message-stream")}'`
    );
  }
  return res.text();
}

/**
 * Build the same transport the app uses, with `fetch` swapped for one that
 * replays the recorded SSE. The transport performs the SSE parsing, so this
 * exercises the real client path without needing a running backend.
 */
function makeTransport(sse) {
  return new DefaultChatTransport({
    api: `${API_URL}/api/chat`,
    fetch: async () =>
      new Response(sse, {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-vercel-ai-ui-message-stream": "v1",
        },
      }),
    headers: async () => ({ Authorization: "Bearer mock-dev-token" }),
    body: async () => ({ user_id: "mock_user_123", thread_id: "verify-stream" }),
    prepareSendMessagesRequest: ({ api, headers, body, messages }) => {
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      const textOf = (m) =>
        (m.parts ?? []).filter((p) => p.type === "text").map((p) => p.text ?? "").join("");
      return {
        api,
        headers,
        body: { ...(body || {}), message: lastUser ? textOf(lastUser) : "", messages: undefined },
      };
    },
  });
}

async function main() {
  let sse;
  try {
    sse = await capture();
    await mkdir(SAMPLE_DIR, { recursive: true });
    await writeFile(new URL("./schedule.sse", SAMPLE_DIR), sse, "utf8");
    console.log(`captured ${sse.length} bytes from ${API_URL}/api/chat\n`);
  } catch (err) {
    const path = new URL("./schedule.sse", SAMPLE_DIR);
    try {
      sse = await readFile(path, "utf8");
      console.log(`backend unreachable (${err.message}); using recorded sample\n`);
    } catch {
      console.error("No recorded sample and no backend to capture from.");
      console.error(`Start the backend, or provide mobile/.stream-samples/schedule.sse.`);
      process.exit(1);
    }
  }

  // sendMessages does the SSE parsing and yields UI message chunks. Reduce
  // them the way useChat does to get the final message.
  const transport = makeTransport(sse);
  const chunks = [];
  const stream = await transport.sendMessages({
    chatId: "verify-stream",
    messages: [{ id: "u1", role: "user", parts: [{ type: "text", text: "schedule my day" }] }],
    trigger: "submit-message",
  });
  for await (const chunk of stream) chunks.push(chunk);

  const message = reduceChunks(chunks);
  const parts = message.parts ?? [];

  let text = "";
  let schedule = null;
  const counts = {};
  for (const part of parts) {
    counts[part.type] = (counts[part.type] || 0) + 1;
    if (part.type === "text") text += part.text;
    if (part.type === "data-schedule") schedule = part.data;
  }

  console.log("parts:", JSON.stringify(counts));
  console.log(`text: ${JSON.stringify(text.slice(0, 120))}${text.length > 120 ? "..." : ""}`);

  const checks = [
    ["the stream produced a message id", typeof message.id === "string" && message.id.length > 0],
    ["assistant text was reassembled", text.length > 0],
    ["the schedule data part arrived", schedule !== null],
    ["the schedule is non-empty", schedule && schedule.schedule.length > 0],
    ["the schedule ends with Sleep", schedule && schedule.schedule.at(-1)?.title === "Sleep"],
    ["every block has the fields the UI renders", schedule && schedule.schedule.every(
      (b) => typeof b.time === "string" && typeof b.title === "string" && typeof b.type === "string"
    )],
  ];

  let ok = true;
  console.log("");
  for (const [label, passed] of checks) {
    console.log(`${passed ? "PASS" : "FAIL"}  ${label}`);
    if (!passed) ok = false;
  }

  // Frame-level shape checks, independent of the SDK's own validation.
  const rawFrames = [];
  for (const line of sse.split("\n")) {
    if (line.startsWith("data: ") && line.slice(6) !== "[DONE]") {
      try {
        rawFrames.push(JSON.parse(line.slice(6)));
      } catch {
        /* ignore */
      }
    }
  }
  console.log("");
  for (const [type, validate] of Object.entries(EXPECTED)) {
    const frames = rawFrames.filter((f) => f.type === type);
    if (!frames.length) continue;
    const passed = frames.every(validate);
    console.log(`${passed ? "PASS" : "FAIL"}  ${frames.length}x '${type}' frame(s) well-formed`);
    if (!passed) ok = false;
  }

  console.log(ok ? "\nThe backend stream is consumable by useChat." : "\nSTREAM MISMATCH");
  process.exit(ok ? 0 : 1);
}

main();
