import { useState, useCallback, useEffect, useRef } from "react";
import { authService } from "../services/firebase";
import { API_URL, authHeaders } from "../services/apiHeaders";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  data?: any;
  createdAt: Date;
}

export function useAgentChat(threadId: string) {
  const [token, setToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>(authService.currentUser?.uid || "mock_user_123");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isApproving, setIsApproving] = useState<boolean>(false);
  const [isOffline, setIsOffline] = useState<boolean>(false);
  const msgIdCounter = useRef(0);

  useEffect(() => {
    const unsubAuth = authService.onAuthStateChanged(async (user: any) => {
      if (user) {
        setUserId(user.uid);
        try {
          const jwt = await authService.getIdToken();
          setToken(jwt);
        } catch (err) {
          setToken("mock-dev-token");
        }
      } else {
        setUserId("mock_user_123");
        setToken("mock-dev-token");
      }
    });
    return unsubAuth;
  }, []);

  const apiUrl = API_URL;

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${apiUrl}/health`, { method: "GET" });
        if (response.ok) {
          setIsOffline(false);
        } else {
          setIsOffline(true);
        }
      } catch (err) {
        setIsOffline(true);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, [apiUrl]);



  const nextId = (prefix: string) => `${prefix}_${Date.now()}_${++msgIdCounter.current}`;

  // ── Send a chat message to the backend and read the SSE stream ──
  const handleSubmit = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    // Append user message immediately
    const userMsg: ChatMessage = {
      id: nextId("user"),
      role: "user",
      content: text,
      createdAt: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: "POST",
        headers: await authHeaders(),
        body: JSON.stringify({
          user_id: userId,
          thread_id: threadId,
          message: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // React Native fetch doesn't support ReadableStream.
      // Read the full SSE response as text and parse line by line.
      const responseText = await response.text();
      let assistantText = "";
      let proposedSchedule: any = null;

      const lines = responseText.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;

        const payload = trimmed.slice(6); // strip "data: "
        if (payload === "[DONE]") continue;

        try {
          const parsed = JSON.parse(payload);
          if (parsed.type === "text" && parsed.text) {
            assistantText += parsed.text;
          } else if (parsed.type === "proposed_schedule" && parsed.schedule) {
            proposedSchedule = parsed.schedule;
          } else if (parsed.type === "approved_schedule") {
            assistantText += assistantText
              ? "\n✅ Schedule saved!"
              : "✅ Schedule saved!";
          } else if (parsed.type === "error") {
            assistantText += `⚠️ Error: ${parsed.message}`;
          }
        } catch {
          // Not valid JSON — skip
        }
      }

      // Append assistant response(s)
      if (assistantText) {
        const assistantMsg: ChatMessage = {
          id: nextId("assistant"),
          role: "assistant",
          content: assistantText,
          createdAt: new Date(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      }

      if (proposedSchedule) {
        const scheduleMsg: ChatMessage = {
          id: nextId("schedule"),
          role: "assistant",
          content: JSON.stringify({ type: "proposed_schedule", schedule: proposedSchedule }),
          data: { schedule: proposedSchedule },
          createdAt: new Date(),
        };
        setMessages((prev) => [...prev, scheduleMsg]);
      }
    } catch (err) {
      console.error("Agent chat error:", err);
      const errorMsg: ChatMessage = {
        id: nextId("error"),
        role: "assistant",
        content: "Sorry, I couldn't connect to the server. Please check that the backend is running.",
        createdAt: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, apiUrl, userId, threadId, token]);

  // ── Approve / edit a proposed schedule ──
  const approveSchedule = useCallback(
    async (feedback: { approved: boolean; edits: any[] }) => {
      setIsApproving(true);
      try {
        const response = await fetch(`${apiUrl}/api/schedule/approve`, {
          method: "POST",
          headers: await authHeaders(),
          body: JSON.stringify({
            thread_id: threadId,
            user_id: userId,
            feedback,
          }),
        });

        const data = await response.json();
        if (data.status === "success") {
          // User feedback message
          let userMsgContent = "Looks good ✓";
          if (!feedback.approved) {
            userMsgContent =
              "Requesting edits:\n" +
              feedback.edits
                .map((e) => {
                  if (e.action === "move") return `• Move "${e.block}" to ${e.new_time}`;
                  if (e.action === "remove") return `• Remove "${e.block}"`;
                  return `• ${e.action} "${e.block}"`;
                })
                .join("\n");
          }

          const userMsg: ChatMessage = {
            id: nextId("user_feedback"),
            role: "user",
            content: userMsgContent,
            createdAt: new Date(),
          };

          const assistantMsg: ChatMessage = {
            id: nextId("assistant_feedback"),
            role: "assistant",
            content: feedback.approved
              ? "Awesome! Your schedule has been saved and reminders are set."
              : "I've rebuilt your schedule with those changes. Here is the new proposal:",
            data: feedback.approved ? undefined : { schedule: data.proposed_schedule },
            createdAt: new Date(),
          };

          setMessages((prev) => [...prev, userMsg, assistantMsg]);
        } else {
          console.error("Approve API failed:", data.message);
        }
      } catch (err) {
        console.error("Error calling approve endpoint:", err);
      } finally {
        setIsApproving(false);
      }
    },
    [apiUrl, threadId, token]
  );

  return {
    messages,
    setMessages,
    input,
    setInput,
    isLoading,
    isApproving,
    isOffline,
    handleSubmit,
    approveSchedule,
  };
}
