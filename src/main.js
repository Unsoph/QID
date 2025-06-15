
import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

async function init() {
  session = await ort.InferenceSession.create('best.onnx');
  console.log("ONNX model loaded.");
}

function preprocessImage(image) {
  const width = 640;
  const height = 640;

  // Resize image for model
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

async function handleImageUpload(event) {
  const file = event.target.files[0];
  const img = new Image();

  img.onload = async () => {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0, img.width, img.height);

  const inputTensor = preprocessImage(img);
  const feeds = { images: inputTensor };

  const output = await session.run(feeds);
  const outputName = Object.keys(output)[0];
  const outputData = output[outputName].data;
  const outputShape = output[outputName].dims; // [1, 5, 8400]

  const numDetections = outputShape[2];
  const inputSize = 640;

  for (let i = 0; i < numDetections; i++) {
    const x = outputData[i * 5];
    const y = outputData[i * 5 + 1];
    const w = outputData[i * 5 + 2];
    const h = outputData[i * 5 + 3];
    const conf = outputData[i * 5 + 4];

    const left = (x - w / 2) * (img.width / inputSize);
    const top = (y - h / 2) * (img.height / inputSize);
    const boxWidth = w * (img.width / inputSize);
    const boxHeight = h * (img.height / inputSize);

    ctx.beginPath();
    ctx.rect(left, top, boxWidth, boxHeight);
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(0,255,0,${conf.toFixed(2)})`; // green box with transparency
    ctx.stroke();

    // Fill red if low confidence
    ctx.fillStyle = `rgba(255,0,0,${(1 - conf).toFixed(2)})`;
    ctx.fillRect(left, top, boxWidth, boxHeight);
  }

  alert("Heatmap drawn. Check image for bounding box confidence levels.");
};


    console.log(`Detections before NMS: ${boxes.length}`);
    const finalBoxes = nonMaxSuppression(boxes);
    console.log(`Detections after NMS: ${finalBoxes.length}`);

    // Redraw original image before drawing boxes
    ctx.drawImage(img, 0, 0, img.width, img.height);

    const scaleX = img.width / 640;
    const scaleY = img.height / 640;

    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'lime';

    for (const box of finalBoxes) {
      const left = (box.x - box.w / 2) * scaleX;
      const top = (box.y - box.h / 2) * scaleY;
      const width = box.w * scaleX;
      const height = box.h * scaleY;

      ctx.strokeRect(left, top, width, height);
      ctx.fillText(`Conf: ${box.conf.toFixed(2)}`, left, top > 20 ? top - 5 : top + 15);
    }
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
