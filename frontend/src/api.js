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

// --- tasks with a wager (Checkpoint 4) ---

async function post(path, body) {
  return json(
    await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  );
}

export function createTask({ userId, description, taskType, wagerCents, deadlineS, nViews, sessionS, nCheckins }) {
  return post("/api/tasks", {
    user_id: userId,
    description,
    task_type: taskType,
    wager_cents: wagerCents,
    deadline_s: deadlineS,
    n_views: nViews || 0,
    session_s: sessionS || 0,
    n_checkins: nCheckins || 0,
  });
}

export async function getTask(taskId, userId) {
  return json(await fetch(`/api/tasks/${taskId}?user_id=${encodeURIComponent(userId)}`));
}

export function startSession(taskId, userId) {
  return post(`/api/tasks/${taskId}/session`, { user_id: userId });
}

export function addCheckin(taskId, userId, captureId) {
  return post(`/api/tasks/${taskId}/checkins`, { user_id: userId, capture_id: captureId });
}

export function submitTaskWithWager({ taskId, userId, beforeCaptureIds, afterCaptureIds }) {
  return post(`/api/tasks/${taskId}/submit`, {
    user_id: userId,
    before_capture_ids: beforeCaptureIds,
    after_capture_ids: afterCaptureIds,
  });
}

// A multi-view capture asks for the angle the task assigned to that view.
export async function requestViewChallenge(userId, role, taskId, viewIndex) {
  return json(
    await fetch("/api/challenges", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, role, task_id: taskId, view_index: viewIndex }),
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
