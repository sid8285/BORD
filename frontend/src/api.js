// Thin client for the FastAPI service (backend/bord/api.py). Every timestamp that matters is
// taken by the server; nothing here sends a client clock or a client filename.

async function json(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `request failed (${response.status})`);
  }
  return body;
}

export async function requestChallenge(userId, role) {
  return json(
    await fetch("/api/challenges", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, role }),
    })
  );
}

export async function uploadCapture({ userId, role, challenge, blob }) {
  const form = new FormData();
  form.append("user_id", userId);
  form.append("role", role);
  form.append("challenge_id", challenge.id);
  form.append("code", challenge.code);
  form.append("photo", blob, "capture.jpg");
  return json(await fetch("/api/captures", { method: "POST", body: form }));
}

export async function submitTask({ userId, taskDescription, beforeCaptureId, afterCaptureId }) {
  return json(
    await fetch("/api/submissions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        task_description: taskDescription,
        before_capture_id: beforeCaptureId,
        after_capture_id: afterCaptureId,
      }),
    })
  );
}

export async function postDeviceReport({ label, blob, captureMs }) {
  const form = new FormData();
  form.append("user_agent", navigator.userAgent);
  form.append("label", label);
  if (captureMs !== undefined) form.append("capture_ms", String(Math.round(captureMs)));
  form.append("photo", blob, "frame.jpg");
  return json(await fetch("/api/device-reports", { method: "POST", body: form }));
}
