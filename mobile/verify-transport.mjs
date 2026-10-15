// Verifies the transport adapter against the real installed `ai` package.
// Run with: node verify-transport.mjs
import { DefaultChatTransport } from "ai";

const API_URL = "http://localhost:8000";

const transport = new DefaultChatTransport({
  api: `${API_URL}/api/chat`,
  headers: async () => ({ Authorization: "Bearer mock-dev-token" }),
  body: async () => ({ user_id: "mock_user_123", thread_id: "default-chat-thread" }),
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

// Reach into the protected fields the same way the SDK's sendMessages does.
const request = await transport.prepareSendMessagesRequest({
  api: transport.api,
  id: "chat-1",
  messages: [
    { id: "m1", role: "user", parts: [{ type: "text", text: "schedule my day" }] },
    {
      id: "m2",
      role: "assistant",
      parts: [
        { type: "text", text: "Here is your plan." },
        { type: "data-schedule", data: { status: "proposed", schedule: [] } },
      ],
    },
  ],
  body: { user_id: "mock_user_123", thread_id: "default-chat-thread" },
  headers: { Authorization: "Bearer mock-dev-token" },
  credentials: undefined,
  requestMetadata: undefined,
  trigger: "submit-message",
  messageId: undefined,
});

const sent = JSON.parse(JSON.stringify(request.body));

const checks = [
  ["message is the newest user turn", sent.message === "schedule my day"],
  ["the SDK messages array is dropped", !("messages" in sent)],
  ["thread_id survives", sent.thread_id === "default-chat-thread"],
  ["user_id survives", sent.user_id === "mock_user_123"],
  ["api is absolute", request.api === `${API_URL}/api/chat`],
  ["auth header survives", request.headers.Authorization === "Bearer mock-dev-token"],
];

let ok = true;
for (const [label, passed] of checks) {
  console.log(`${passed ? "PASS" : "FAIL"}  ${label}`);
  if (!passed) ok = false;
}
console.log(`\nbody sent: ${JSON.stringify(sent)}`);
console.log(ok ? "\nAll transport checks passed." : "\nSOME CHECKS FAILED");
process.exit(ok ? 0 : 1);
