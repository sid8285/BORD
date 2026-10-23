import { useCallback, useEffect, useRef, useState } from "react";
import { requestChallenge, uploadCapture } from "./api.js";
import { grabFrame, openCamera, stopCamera } from "./camera.js";

// One capture step ("before" or "after"). Shows the live camera and the challenge the server
// issued; on Capture it draws the code into the frame and uploads it at once, so the server's
// receive time is the capture time. The challenge lifetime is counted down locally from the
// moment it arrived (relative time only; the server decides whether it is still valid).
export default function CameraCapture({ userId, role, onCaptured }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [challenge, setChallenge] = useState(null);
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [status, setStatus] = useState("starting");
  const [error, setError] = useState("");

  const newChallenge = useCallback(async () => {
    const c = await requestChallenge(userId, role);
    setChallenge(c);
    setSecondsLeft(Math.round(c.expires_at - c.issued_at));
    return c;
  }, [userId, role]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = await openCamera();
        if (cancelled) return stopCamera(stream);
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await newChallenge();
        setStatus("ready");
      } catch (e) {
        setError(e.message);
        setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
      stopCamera(streamRef.current);
    };
  }, [newChallenge]);

  useEffect(() => {
    if (status !== "ready") return undefined;
    const timer = setInterval(() => setSecondsLeft((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(timer);
  }, [status, challenge]);

  async function capture() {
    setStatus("uploading");
    setError("");
    try {
      const { blob } = await grabFrame(videoRef.current, challenge);
      const result = await uploadCapture({ userId, role, challenge, blob });
      if (!result.challenge_ok) {
        // Expired or already used: get a fresh code and let the user retake.
        setError(`The server rejected that capture (${result.reason}). Take it again.`);
        await newChallenge();
        setStatus("ready");
        return;
      }
      onCaptured(result.capture_id);
    } catch (e) {
      setError(e.message);
      setStatus("ready");
    }
  }

  return (
    <section className="capture">
      <h2>{role === "before" ? "Before photo" : "After photo"}</h2>
      <video ref={videoRef} autoPlay playsInline muted />
      {challenge && (
        <p className="challenge">
          Code <strong>{challenge.code}</strong> will be stamped on the photo. Hold the camera{" "}
          <strong>{challenge.angle}</strong>. Valid for {secondsLeft}s.
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <button onClick={capture} disabled={status !== "ready" || secondsLeft === 0}>
        {status === "uploading" ? "Uploading..." : "Capture"}
      </button>
      {secondsLeft === 0 && status === "ready" && (
        <button onClick={newChallenge}>Get a new code</button>
      )}
    </section>
  );
}
