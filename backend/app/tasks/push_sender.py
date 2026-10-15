"""Background job that sends pending push notifications via the Expo Push API.

Runs every 60 seconds via APScheduler. For each user with pending notifications
where scheduled_for <= now and sent == false, it sends the notification via
Expo's push service and marks it as sent in Firestore.
"""

import logging
import datetime
from typing import List, Dict, Any

import httpx
from firebase_admin import firestore as firestore_sync

logger = logging.getLogger("app.tasks.push_sender")

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def _get_firestore_client():
    """Get synchronous Firestore client (APScheduler jobs run in threads)."""
    return firestore_sync.client()


def _build_push_message(push_token: str, title: str, body: str) -> Dict[str, Any]:
    """Build an Expo push notification message payload."""
    return {
        "to": push_token,
        "title": title,
        "body": body,
        "sound": "default",
        "channelId": "schedule-reminders",
        "data": {"type": "schedule_reminder"},
    }


def send_pending_notifications():
    """Main job function — called by APScheduler every minute.

    1. Lists all users that have a push_token in their Firestore user doc.
    2. For each user, checks notifications/{uid}/pending/{today} for unsent reminders.
    3. Sends each due reminder via Expo Push API.
    4. Marks sent=true after successful delivery.
    """
    try:
        from app.api.auth import firebase_initialized
        if not firebase_initialized:
            logger.warning("Push sender job: Firebase not initialized. Skipping push check.")
            return
    except Exception as e:
        logger.error(f"Push sender job: could not import/check Firebase initialization status: {e}")
        return

    logger.info("Push sender job: checking for pending notifications...")

    try:
        db = _get_firestore_client()
        today = datetime.date.today().isoformat()
        now = datetime.datetime.now()

        # Get all user documents that have a push_token
        users_ref = db.collection("users")
        users_stream = users_ref.stream()

        sent_count = 0
        messages_to_send: List[Dict[str, Any]] = []
        update_refs: List[Any] = []

        for user_doc in users_stream:
            user_data = user_doc.to_dict() or {}
            push_token = user_data.get("push_token")
            if not push_token:
                continue

            uid = user_doc.id

            # Check pending notifications for today
            pending_ref = db.collection("notifications").document(uid).collection("pending").document(today)
            pending_doc = pending_ref.get()

            if not pending_doc.exists:
                continue

            pending_data = pending_doc.to_dict() or {}
            reminders = pending_data.get("reminders", [])
            updated = False

            for i, reminder in enumerate(reminders):
                if reminder.get("sent", False):
                    continue

                # Parse scheduled_for or fall back to time field
                scheduled_for = reminder.get("scheduled_for")
                if scheduled_for:
                    if isinstance(scheduled_for, str):
                        try:
                            trigger_time = datetime.datetime.fromisoformat(scheduled_for)
                        except ValueError:
                            continue
                    else:
                        trigger_time = scheduled_for
                else:
                    # Fall back to time field (HH:MM) + today's date
                    time_str = reminder.get("time", "")
                    if not time_str:
                        continue
                    try:
                        hour, minute = map(int, time_str.split(":"))
                        trigger_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    except (ValueError, TypeError):
                        continue

                # Check if it's time to send
                if trigger_time <= now:
                    title = reminder.get("title", "Schedule Reminder")
                    body = reminder.get("body", f"Time for: {title}")
                    messages_to_send.append(_build_push_message(push_token, title, body))
                    update_refs.append((pending_ref, i, reminders))
                    reminders[i]["sent"] = True
                    updated = True
                    sent_count += 1

            if updated:
                pending_ref.update({"reminders": reminders})

        # Batch send via Expo Push API
        if messages_to_send:
            _send_expo_push_batch(messages_to_send)

        if sent_count > 0:
            logger.info(f"Push sender job: sent {sent_count} notification(s).")
        else:
            logger.debug("Push sender job: no pending notifications due.")

    except Exception as e:
        logger.error(f"Push sender job error: {e}", exc_info=True)


def _send_expo_push_batch(messages: List[Dict[str, Any]]):
    """Send a batch of push messages to the Expo Push API (synchronous httpx)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            # Expo accepts up to 100 messages per request
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                resp = client.post(
                    EXPO_PUSH_URL,
                    json=batch,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    errors = [t for t in data.get("data", []) if t.get("status") == "error"]
                    if errors:
                        logger.warning(f"Expo push errors: {errors}")
                    else:
                        logger.info(f"Expo push batch sent successfully ({len(batch)} messages).")
                else:
                    logger.error(f"Expo push API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed to send Expo push batch: {e}", exc_info=True)
