import * as ort from 'onnxruntime-web';
import EXIF from 'exif-js';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

// Initialize ONNX model
async function init() {
  session = await ort.InferenceSession.create('./best.onnx');
  console.log("ONNX model loaded.");
}

function preprocessImage(image) {
  const width = 640;
  const height = 640;

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;

  const input = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    input[i] = data[i * 4] / 255; // R
    input[i + width * height] = data[i * 4 + 1] / 255; // G
    input[i + 2 * width * height] = data[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor('float32', input, [1, 3, height, width]);
}

// Fix orientation using EXIF
function fixOrientation(image, orientation, callback) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const w = image.width;
  const h = image.height;

  if ([5,6,7,8].includes(orientation)) {
    canvas.width = h;
    canvas.height = w;
  } else {
    canvas.width = w;
    canvas.height = h;
  }

  switch (orientation) {
    case 2: ctx.transform(-1, 0, 0, 1, w, 0); break;
    case 3: ctx.transform(-1, 0, 0, -1, w, h); break;
    case 4: ctx.transform(1, 0, 0, -1, 0, h); break;
    case 5: ctx.transform(0, 1, 1, 0, 0, 0); break;
    case 6: ctx.transform(0, 1, -1, 0, h, 0); break;
    case 7: ctx.transform(0, -1, -1, 0, h, w); break;
    case 8: ctx.transform(0, -1, 1, 0, 0, w); break;
    default: ctx.transform(1, 0, 0, 1, 0, 0);
  }

  ctx.drawImage(image, 0, 0);
  const correctedImage = new Image();
  correctedImage.onload = () => callback(correctedImage);
  correctedImage.src = canvas.toDataURL();
}

function nonMaxSuppression(boxes, iouThreshold = 0.5) {
  boxes.sort((a, b) => b.conf - a.conf);
  const selected = [];

  for (let i = 0; i < boxes.length; i++) {
    const a = boxes[i];
    let keep = true;

    for (let j = 0; j < selected.length; j++) {
      const b = selected[j];
      const iou = calculateIoU(a, b);
      if (iou > iouThreshold) {
        keep = false;
        break;
      }
    }

    if (keep) selected.push(a);
  }

  return selected;
}

function calculateIoU(a, b) {
  const x1 = Math.max(a.x - a.w / 2, b.x - b.w / 2);
  const y1 = Math.max(a.y - a.h / 2, b.y - b.h / 2);
  const x2 = Math.min(a.x + a.w / 2, b.x + b.w / 2);
  const y2 = Math.min(a.y + a.h / 2, b.y + b.h / 2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const boxAArea = a.w * a.h;
  const boxBArea = b.w * b.h;

  return interArea / (boxAArea + boxBArea - interArea);
}

async function detectAndDraw(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const inputTensor = preprocessImage(img);
  const feeds = { images: inputTensor };

  const output = await session.run(feeds);
  const rawData = output[Object.keys(output)[0]].data;
  const [batch, channels, numDetections] = output[Object.keys(output)[0]].dims;

  const boxes = [];

  for (let i = 0; i < numDetections; i++) {
    const x = rawData[i];
    const y = rawData[i + numDetections];
    const w = rawData[i + 2 * numDetections];
    const h = rawData[i + 3 * numDetections];
    const conf = rawData[i + 4 * numDetections];

    if (conf > 0.4) {
      boxes.push({ x, y, w, h, conf });
    }
  }

  const finalBoxes = nonMaxSuppression(boxes);

  // Draw + crop
  finalBoxes.forEach((box, idx) => {
    const scaleX = img.width / 640;
    const scaleY = img.height / 640;

    const left = (box.x - box.w / 2) * scaleX;
    const top = (box.y - box.h / 2) * scaleY;
    const width = box.w * scaleX;
    const height = box.h * scaleY;

    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.strokeRect(left, top, width, height);

    // Crop
    const cropped = document.createElement('canvas');
    cropped.width = width;
    cropped.height = height;
    cropped.getContext('2d').drawImage(canvas, left, top, width, height, 0, 0, width, height);
    document.body.appendChild(cropped); // For demo; can also upload or save
  });
}

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      EXIF.getData(img, function () {
        const orientation = EXIF.getTag(this, "Orientation") || 1;
        fixOrientation(img, orientation, corrected => {
          detectAndDraw(corrected);
        });
      });
    };
    img.src = reader.result;
  };

  reader.readAsDataURL(file);
});

document.addEventListener('DOMContentLoaded', init);