import { authService } from "./firebase";

export const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Authorization headers for backend calls.
 *
 * The backend verifies a Firebase ID token on every endpoint. When Firebase is
 * not configured (the mock auth used for local dev), `authService` returns a
 * placeholder token and the backend runs in its documented dev fallback, so the
 * demo flow keeps working.
 */
export async function authHeaders(
  extra?: Record<string, string>
): Promise<Record<string, string>> {
  let token = "mock-dev-token";
  try {
    token = (await authService.getIdToken()) || "mock-dev-token";
  } catch {
    // keep the dev fallback
  }
  return { ...(extra || {}), Authorization: `Bearer ${token}` };
}
