import { useState } from "react";
import CameraCapture from "./CameraCapture.jsx";
import DeviceTest from "./DeviceTest.jsx";
import { submitTask } from "./api.js";

const USER_KEY = "bord-user-id";

// A stand-in for sign-in until Checkpoint 4 adds accounts: a random id kept in this browser.
function localUserId() {
  let id = null;
  try {
    id = localStorage.getItem(USER_KEY);
  } catch (e) {
    id = null;
  }
  if (!id) {
    id = "u" + Math.random().toString(36).slice(2, 10);
    try {
      localStorage.setItem(USER_KEY, id);
    } catch (e) {
      // private mode: the id lasts for this page load only
    }
  }
  return id;
}

export default function App() {
  const [userId] = useState(localUserId);
  const [task, setTask] = useState("");
  const [step, setStep] = useState("task"); // task -> before -> after -> verifying -> result
  const [beforeId, setBeforeId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  if (new URLSearchParams(window.location.search).has("device-test")) return <DeviceTest />;

  async function finish(afterId) {
    setStep("verifying");
    setError("");
    try {
      setResult(
        await submitTask({ userId, taskDescription: task, beforeCaptureId: beforeId, afterCaptureId: afterId })
      );
      setStep("result");
    } catch (e) {
      setError(e.message);
      setStep("result");
    }
  }

  return (
    <main>
      <h1>BORD</h1>
      {step === "task" && (
        <section>
          <label>
            What will you do?
            <input value={task} onChange={(e) => setTask(e.target.value)} placeholder="Put the dishes away" />
          </label>
          <button disabled={!task.trim()} onClick={() => setStep("before")}>
            Take the before photo
          </button>
        </section>
      )}
      {step === "before" && (
        <CameraCapture
          key="before"
          userId={userId}
          role="before"
          onCaptured={(id) => {
            setBeforeId(id);
            setStep("after");
          }}
        />
      )}
      {step === "after" && (
        <>
          <p>Task: {task}. Do it, then take the after photo.</p>
          <CameraCapture key="after" userId={userId} role="after" onCaptured={finish} />
        </>
      )}
      {step === "verifying" && <p>Checking your photos...</p>}
      {step === "result" && (
        <section>
          {error && <p className="error">{error}</p>}
          {result && (
            <>
              <h2>
                {result.decision === "release" && "Verified: your wager is returned."}
                {result.decision === "forfeit" && "Not verified: the wager goes to charity."}
                {result.decision === "hold" && "On hold: a person will review your photos."}
              </h2>
              <p>{result.reason}</p>
              {result.flagged_by.length > 0 && <p>Flagged by: {result.flagged_by.join(", ")}</p>}
              <p>
                Server time {result.server_s.toFixed(1)}s{result.deadline_missed ? " (missed the deadline)" : ""}.
              </p>
            </>
          )}
          <button
            onClick={() => {
              setTask("");
              setResult(null);
              setBeforeId(null);
              setStep("task");
            }}
          >
            Start over
          </button>
        </section>
      )}
    </main>
  );
}
