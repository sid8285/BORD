import { useEffect, useRef, useState } from "react";
import { postDeviceReport } from "./api.js";
import { grabFrame, openCamera, stopCamera } from "./camera.js";

// Assumption 3 device test (open the app with ?device-test on each phone and browser):
// "getUserMedia works on mobile Chrome and Safari, but captured frames carry no EXIF".
// The page opens the camera, captures one frame the same way the real capture page does and
// posts it; the server inspects the bytes and appends a line to data/device_tests.jsonl.
export default function DeviceTest() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [label, setLabel] = useState("");
  const [state, setState] = useState("starting");
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    openCamera()
      .then((stream) => {
        if (cancelled) return stopCamera(stream);
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        setState("ready");
      })
      .catch((e) => {
        setError(e.message);
        setState("error");
      });
    return () => {
      cancelled = true;
      stopCamera(streamRef.current);
    };
  }, []);

  async function run() {
    setState("uploading");
    setError("");
    try {
      // Time the in-browser capture stage the way the real capture page runs it: draw the
      // frame, overlay a challenge code and its stamp strip, encode the JPEG (budget: under 1 s).
      const t0 = performance.now();
      const { blob } = await grabFrame(videoRef.current, { code: "TEST23" });
      const captureMs = performance.now() - t0;
      const t1 = performance.now();
      const result = await postDeviceReport({ label, blob, captureMs });
      setReport({ ...result, upload_ms: Math.round(performance.now() - t1) });
      setState("done");
    } catch (e) {
      setError(e.message);
      setState("ready");
    }
  }

  return (
    <section className="capture">
      <h2>Device test</h2>
      <p>{navigator.userAgent}</p>
      <video ref={videoRef} autoPlay playsInline muted />
      <input
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        placeholder="Device and browser, e.g. iPhone 16 Pro / Safari"
      />
      {error && <p className="error">{error}</p>}
      <button onClick={run} disabled={state !== "ready"}>
        Capture and report
      </button>
      {report && (
        <pre>
          {JSON.stringify(
            { format: report.format, size: report.size, exif_tag_count: report.exif_tag_count, carries_exif: report.carries_exif, capture_ms: report.capture_ms, upload_ms: report.upload_ms },
            null,
            2
          )}
        </pre>
      )}
    </section>
  );
}
