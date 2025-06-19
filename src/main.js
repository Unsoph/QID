import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');

let cropperSession;
let orientationSession;

async function init() {
  cropperSession = await ort.InferenceSession.create('./best.onnx');
  orientationSession = await ort.InferenceSession.create('./aadhaar_orientation.onnx');
  console.log("✅ Both ONNX models loaded.");
}

function preprocessOrientation(image) {
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d');

  canvasTemp.width = 224;
  canvasTemp.height = 224;
  ctxTemp.drawImage(image, 0, 0, 224, 224);
  const imageData = ctxTemp.getImageData(0, 0, 224, 224).data;

  const floatData = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    floatData[i] = imageData[i * 4] / 255;
    floatData[i + 224 * 224] = imageData[i * 4 + 1] / 255;
    floatData[i + 2 * 224 * 224] = imageData[i * 4 + 2] / 255;
  }

  return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

function rotateImage(image, angle) {
  const canvasRotated = document.createElement('canvas');
  const ctxRotated = canvasRotated.getContext('2d');

  if (angle === 90 || angle === 270) {
    canvasRotated.width = image.height;
    canvasRotated.height = image.width;
  } else {
    canvasRotated.width = image.width;
    canvasRotated.height = image.height;
  }

  ctxRotated.translate(canvasRotated.width / 2, canvasRotated.height / 2);
  ctxRotated.rotate((angle * Math.PI) / 180);
  ctxRotated.drawImage(image, -image.width / 2, -image.height / 2);

  return canvasRotated;
}

function preprocessImage(image) {
  const modelSize = 640;
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d', { willReadFrequently: true });

  const imgW = image.naturalWidth || image.width;
  const imgH = image.naturalHeight || image.height;
  const scale = Math.min(modelSize / imgW, modelSize / imgH);
  const resizedW = Math.round(imgW * scale);
  const resizedH = Math.round(imgH * scale);

  const padX = Math.floor((modelSize - resizedW) / 2);
  const padY = Math.floor((modelSize - resizedH) / 2);

  canvasTemp.width = modelSize;
  canvasTemp.height = modelSize;

  ctxTemp.fillStyle = 'black';
  ctxTemp.fillRect(0, 0, modelSize, modelSize);
  ctxTemp.drawImage(image, padX, padY, resizedW, resizedH);

  const imageData = ctxTemp.getImageData(0, 0, modelSize, modelSize).data;
  const floatData = new Float32Array(modelSize * modelSize * 3);

  for (let i = 0; i < modelSize * modelSize; i++) {
    floatData[i] = imageData[i * 4] / 255;
    floatData[i + modelSize * modelSize] = imageData[i * 4 + 1] / 255;
    floatData[i + 2 * modelSize * modelSize] = imageData[i * 4 + 2] / 255;
  }

  preprocessImage.lastPadX = padX;
  preprocessImage.lastPadY = padY;
  preprocessImage.lastScale = scale;

  return new ort.Tensor('float32', floatData, [1, 3, modelSize, modelSize]);
}

function nonMaxSuppression(boxes, iouThreshold = 0.8) {
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

function cropAndDownload(image, x, y, w, h, index) {
  const cropCanvas = document.createElement('canvas');
  const cropCtx = cropCanvas.getContext('2d');
  cropCanvas.width = w;
  cropCanvas.height = h;
  cropCtx.drawImage(image, x, y, w, h, 0, 0, w, h);

  cropCanvas.toBlob(blob => {
    const link = document.createElement('a');
    link.download = `crop_${index}.png`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  }, 'image/png');
}

async function handleImageUpload(event) {
  const file = event.target.files[0];
  const img = new Image();

  img.onload = async () => {
    // Step 1: Predict orientation
    const orientationInput = preprocessOrientation(img);
    const orientationOutput = await orientationSession.run({ input: orientationInput });
    const rotation = orientationOutput.output.data.indexOf(Math.max(...orientationOutput.output.data)) * 90;

    // Step 2: Rotate to correct orientation
    const correctedCanvas = rotateImage(img, rotation);

    // Step 3: Draw corrected image on main canvas
    canvas.width = correctedCanvas.width;
    canvas.height = correctedCanvas.height;
    ctx.drawImage(correctedCanvas, 0, 0);

    // Step 4: Run cropper model
    const inputTensor = preprocessImage(correctedCanvas);
    const feeds = { images: inputTensor };
    const output = await cropperSession.run(feeds);
    const outputTensor = output[Object.keys(output)[0]];
    const rawData = outputTensor.data;
    const [batch, channels, numDetections] = outputTensor.dims;

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

    ctx.drawImage(correctedCanvas, 0, 0);

    const scale = 1 / preprocessImage.lastScale;
    const padX = preprocessImage.lastPadX;
    const padY = preprocessImage.lastPadY;

    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'lime';

    for (let i = 0; i < finalBoxes.length; i++) {
      const box = finalBoxes[i];
      const x = (box.x - padX) * scale;
      const y = (box.y - padY) * scale;
      const w = box.w * scale;
      const h = box.h * scale;
      const left = x - w / 2;
      const top = y - h / 2;

      ctx.strokeRect(left, top, w, h);
      ctx.fillText(`Conf: ${box.conf.toFixed(2)}`, left, top > 20 ? top - 5 : top + 15);

      cropAndDownload(correctedCanvas, left, top, w, h, i + 1);
    }
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
