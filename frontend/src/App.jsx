import { useState } from "react";
import CameraCapture from "./CameraCapture.jsx";
import DeviceTest from "./DeviceTest.jsx";
import { createTask, submitTaskWithWager } from "./api.js";

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
  const [wager, setWager] = useState("5.00"); // dollars, Stripe test mode: no real money moves
  const [hours, setHours] = useState("1");    // deadline, hours from now (server clock decides)
  const [taskId, setTaskId] = useState(null);
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
        await submitTaskWithWager({
          taskId,
          userId,
          beforeCaptureIds: [beforeId],
          afterCaptureIds: [afterId],
        })
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
          <label>
            Wager (USD, test mode)
            <input value={wager} onChange={(e) => setWager(e.target.value)} inputMode="decimal" />
          </label>
          <label>
            Deadline (hours from now, up to 168)
            <input value={hours} onChange={(e) => setHours(e.target.value)} inputMode="decimal" />
          </label>
          {error && <p className="error">{error}</p>}
          <button
            disabled={!task.trim() || !(parseFloat(wager) >= 0.5) || !(parseFloat(hours) > 0)}
            onClick={async () => {
              setError("");
              try {
                // The server places the Stripe hold before the first photo is taken.
                const created = await createTask({
                  userId,
                  description: task,
                  taskType: "single_scene",
                  wagerCents: Math.round(parseFloat(wager) * 100),
                  deadlineS: parseFloat(hours) * 3600,
                });
                setTaskId(created.task.id);
                setStep("before");
              } catch (e) {
                setError(e.message);
              }
            }}
          >
            Hold the wager and take the before photo
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
              {result.pairs.some((p) => p.flagged_by.length > 0) && (
                <p>Flagged by: {[...new Set(result.pairs.flatMap((p) => p.flagged_by))].join(", ")}</p>
              )}
              <p>
                Server time {result.server_s.toFixed(1)}s
                {result.pairs.some((p) => p.deadline_missed) ? " (missed the deadline)" : ""}.
              </p>
            </>
          )}
          <button
            onClick={() => {
              setTask("");
              setTaskId(null);
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
