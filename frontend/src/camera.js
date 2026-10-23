// Camera-only capture helpers. There is deliberately no <input type="file"> anywhere in the app:
// a photo can only come from a live getUserMedia stream drawn onto a canvas.

export async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot open the camera (needs https or localhost).");
  }
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } },
    audio: false,
  });
}

export function stopCamera(stream) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
}

// Draw the challenge code into the pixels of the captured frame, so the code travels with the
// photo and a photo taken before the challenge was issued cannot show it.
export function drawChallengeOverlay(ctx, width, height, challenge) {
  const pad = Math.round(width * 0.02);
  const size = Math.round(width * 0.06);
  ctx.save();
  ctx.font = `bold ${size}px ui-monospace, Menlo, monospace`;
  ctx.textBaseline = "top";
  const text = challenge.code;
  const boxWidth = ctx.measureText(text).width + pad * 2;
  const boxHeight = size + pad * 2;
  ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
  ctx.fillRect(pad, pad, boxWidth, boxHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, pad * 2, pad * 2);
  ctx.restore();
}

// Grab the current video frame as a JPEG blob. canvas.toBlob drops all EXIF (Assumption 3).
export function grabFrame(video, challenge) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new Error("The camera is not ready yet.");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);
  if (challenge) drawChallengeOverlay(ctx, width, height, challenge);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve({ blob, width, height }) : reject(new Error("Could not encode the frame."))),
      "image/jpeg",
      0.9
    );
  });
}
