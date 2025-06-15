
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

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;

  const input = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    input[i] = data[i * 4] / 255;
    input[i + width * height] = data[i * 4 + 1] / 255;
    input[i + 2 * width * height] = data[i * 4 + 2] / 255;
  }

  return new ort.Tensor('float32', input, [1, 3, height, width]);
}

function nonMaxSuppression(boxes, iouThreshold = 0.8) {
  if (boxes.length === 0) return [];

  boxes.sort((a, b) => b.conf - a.conf);
  const selectedBoxes = [];

  function iou(boxA, boxB) {
    const x1 = Math.max(boxA.x - boxA.w / 2, boxB.x - boxB.w / 2);
    const y1 = Math.max(boxA.y - boxA.h / 2, boxB.y - boxB.h / 2);
    const x2 = Math.min(boxA.x + boxA.w / 2, boxB.x + boxB.w / 2);
    const y2 = Math.min(boxA.y + boxA.h / 2, boxB.y + boxB.h / 2);

    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;
    const union = boxAArea + boxBArea - interArea;

    return interArea / union;
  }

  while (boxes.length > 0) {
    const chosen = boxes.shift();
    selectedBoxes.push(chosen);
    boxes = boxes.filter(box => iou(chosen, box) < iouThreshold);
  }

  return selectedBoxes;
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
    const outputTensor = output[Object.keys(output)[0]];
    const [batch, numPreds, numAttrs] = outputTensor.dims;

    const data = outputTensor.data;
    const boxes = [];

    for (let i = 0; i < numPreds; i++) {
      const offset = i * numAttrs;
      const [x, y, w, h, conf] = data.slice(offset, offset + 5);

      if (conf > 0.8) {
        boxes.push({ x: x * img.width, y: y * img.height, w: w * img.width, h: h * img.height, conf });
      }
    }

    const finalBoxes = nonMaxSuppression(boxes);

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    for (const box of finalBoxes) {
      const left = box.x - box.w / 2;
      const top = box.y - box.h / 2;
      ctx.strokeRect(left, top, box.w, box.h);
    }

    alert(`Inference complete! ${finalBoxes.length} object(s) detected.`);
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
