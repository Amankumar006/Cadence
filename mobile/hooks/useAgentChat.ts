import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetch as expoFetch } from "expo/fetch";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { z } from "zod";
import { authService } from "../services/firebase";
import { API_URL, authHeaders } from "../services/apiHeaders";

/**
 * Chat transport for the Cadence backend.
 *
 * Three things make this more than a one-liner:
 *
 * 1. React Native's `fetch` does not support streaming responses, so the AI
 *    SDK would buffer the whole SSE stream before handing anything to the UI.
 *    `expo/fetch` does, and it is what makes token-by-token rendering work.
 * 2. The app is not served from the backend's origin, so the transport needs
 *    an absolute URL rather than the SDK's default relative `/api/chat`.
 * 3. Every backend endpoint requires a Firebase ID token, which is fetched
 *    asynchronously per request — hence the resolvable `headers` and `body`.
 */
const scheduleBlockSchema = z
  .object({
    time: z.string(),
    title: z.string(),
    type: z.string(),
    duration_min: z.number(),
    notes: z.string(),
  })
  .passthrough();

/** A `data-schedule` part: either a proposal awaiting approval, or a commit. */
export const schedulePartSchema = z
  .object({
    status: z.enum(["proposed", "approved"]),
    schedule: z.array(scheduleBlockSchema),
  })
  .passthrough();

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  data?: { schedule?: any[]; status?: string };
  createdAt: Date;
}

/** The chat hook's public shape, kept stable so screens do not have to change. */
export interface UseAgentChat {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  isLoading: boolean;
  isApproving: boolean;
  isOffline: boolean;
  handleSubmit: () => void;
  approveSchedule: (feedback: { approved: boolean; edits: any[] }) => Promise<void>;
}

const DEV_USER_ID = "mock_user_123";

/** Flatten a UIMessage's text parts into the plain string the backend wants. */
function textOf(message: { parts?: any[] }): string {
  return (message.parts ?? [])
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join("");
}

export function useAgentChat(threadId: string): UseAgentChat {
  const [userId, setUserId] = useState<string>(
    authService.currentUser?.uid || DEV_USER_ID
  );
  const [isOffline, setIsOffline] = useState<boolean>(false);
  const [isApproving, setIsApproving] = useState<boolean>(false);
  const [input, setInput] = useState<string>("");
  // Messages the client owns because the backend has no endpoint that produces
  // them: the local echo of an approval, and its acknowledgement.
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([]);

  // The transport resolves headers and body per request, so it reads the uid
  // through a ref rather than capturing a stale one from the render it was
  // created in.
  const userIdRef = useRef(userId);
  userIdRef.current = userId;

  useEffect(() => {
    const unsubAuth = authService.onAuthStateChanged((user: any) => {
      setUserId(user?.uid || DEV_USER_ID);
    });
    return unsubAuth;
  }, []);

  // ── Backend reachability ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const checkConnection = async () => {
      try {
        const response = await expoFetch(`${API_URL}/health`, { method: "GET" });
        if (!cancelled) setIsOffline(!response.ok);
      } catch {
        if (!cancelled) setIsOffline(true);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: `${API_URL}/api/chat`,
        fetch: expoFetch,
        credentials: "omit",
        headers: async () =>
          await authHeaders({ "Content-Type": "application/json" }),
        body: async () => ({ user_id: userIdRef.current, thread_id: threadId }),
        // The SDK posts { id, messages, trigger, messageId }. The backend takes
        // a single `message` string and rebuilds history from the LangGraph
        // thread, so the array is dropped and the newest user turn extracted.
        prepareSendMessagesRequest: ({ api, headers, body, messages }) => {
          const lastUser = [...messages].reverse().find((m) => m.role === "user");
          return {
            api,
            headers,
            body: {
              ...(body || {}),
              message: lastUser ? textOf(lastUser) : "",
              messages: undefined,
            },
          };
        },
      }),
    [threadId]
  );

  const chat = useChat({
    id: threadId,
    transport,
    dataPartSchemas: { schedule: schedulePartSchema },
    onError: (error: Error) => {
      console.error("Agent chat error:", error);
    },
  });

  // ── UIMessage[] -> ChatMessage[] ────────────────────────────────────────
  // Kept as a projection so the existing message renderer keeps working. Text
  // parts concatenate into `content`; the typed schedule data part becomes
  // `data.schedule`, which is what the proposal UI reads.
  const serverMessages = useMemo<ChatMessage[]>(
    () =>
      chat.messages.map((m) => {
        const parts = (m.parts ?? []) as any[];
        const content = parts
          .filter((p) => p.type === "text")
          .map((p) => p.text ?? "")
          .join("");

        const schedulePart = parts.find((p) => p.type === "data-schedule");
        const scheduleData = schedulePart?.data;

        return {
          id: m.id,
          role: m.role === "user" ? "user" : "assistant",
          content,
          data: scheduleData
            ? { schedule: scheduleData.schedule, status: scheduleData.status }
            : undefined,
          createdAt: new Date(),
        };
      }),
    [chat.messages]
  );

  const messages = useMemo(
    () => [...serverMessages, ...localMessages],
    [serverMessages, localMessages]
  );

  const handleSubmit = useCallback(() => {
    const text = input.trim();
    if (!text || chat.status === "submitted" || chat.status === "streaming") return;
    setLocalMessages([]);
    setInput("");
    chat.sendMessage({ text });
  }, [input, chat]);

  // ── Approve / edit a proposed schedule ──────────────────────────────────
  const approveSchedule = useCallback(
    async (feedback: { approved: boolean; edits: any[] }) => {
      setIsApproving(true);
      try {
        const response = await expoFetch(`${API_URL}/api/schedule/approve`, {
          method: "POST",
          headers: await authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            thread_id: threadId,
            user_id: userIdRef.current,
            feedback,
          }),
        });

        const data = await response.json();
        if (data.status !== "success") {
          console.error("Approve API failed:", data.message);
          return;
        }

        // Echo the decision locally: approving resumes the graph rather than
        // starting a new turn, so no server message is produced for it.
        const userContent = feedback.approved
          ? "Looks good ✓"
          : "Requesting edits:\n" +
            feedback.edits
              .map((e) => {
                if (e.action === "move") return `• Move "${e.block}" to ${e.new_time}`;
                if (e.action === "remove") return `• Remove "${e.block}"`;
                return `• ${e.action} "${e.block}"`;
              })
              .join("\n");

        setLocalMessages((prev) => [
          ...prev,
          {
            id: `user_feedback_${Date.now()}`,
            role: "user",
            content: userContent,
            createdAt: new Date(),
          },
          {
            id: `assistant_feedback_${Date.now()}`,
            role: "assistant",
            content: feedback.approved
              ? "Awesome! Your schedule has been saved and reminders are set."
              : "I've rebuilt your schedule with those changes.",
            createdAt: new Date(),
          },
        ]);
      } catch (err) {
        console.error("Error calling approve endpoint:", err);
      } finally {
        setIsApproving(false);
      }
    },
    [threadId]
  );

  return {
    messages,
    setMessages: setLocalMessages,
    input,
    setInput,
    isLoading: chat.status === "submitted" || chat.status === "streaming",
    isApproving,
    isOffline,
    handleSubmit,
    approveSchedule,
  };
}
