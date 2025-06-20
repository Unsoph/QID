import * as ort from 'onnxruntime-web';
import JSZip from 'jszip';
import { createWorker } from 'tesseract.js';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const rotationDisplay = document.getElementById('rotation-info');
const loader = document.getElementById('loader');

let cropperSession;

async function init() {
  cropperSession = await ort.InferenceSession.create('./best.onnx');
  console.log("✅ Cropper model loaded.");
}

function rotateImage(image, angle) {
  const canvasRotated = document.createElement('canvas');
  const ctxRotated = canvasRotated.getContext('2d');
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;

  if (angle === 90 || angle === 270) {
    canvasRotated.width = height;
    canvasRotated.height = width;
  } else {
    canvasRotated.width = width;
    canvasRotated.height = height;
  }

  ctxRotated.translate(canvasRotated.width / 2, canvasRotated.height / 2);
  ctxRotated.rotate((angle * Math.PI) / 180);
  ctxRotated.drawImage(image, -width / 2, -height / 2);

  return canvasRotated;
}

async function detectBestRotation(image) {
  const worker = await createWorker({ langPath: 'https://tessdata.projectnaptha.com/4.0.0' });

  await worker.load();
  await worker.loadLanguage('eng+hin');
  await worker.initialize('eng+hin');

  const angles = [0, 90, 180, 270];
  const keywords = ['भारत', 'Government', 'भारत सरकार', 'GOVT OF INDIA', 'Aadhaar'];

  let bestAngle = 0;
  let bestScore = -1;

  for (let angle of angles) {
    const rotated = rotateImage(image, angle);

    const scaledCanvas = document.createElement('canvas');
    const ctx = scaledCanvas.getContext('2d');
    const targetWidth = 600;
    const scale = targetWidth / rotated.width;
    scaledCanvas.width = targetWidth;
    scaledCanvas.height = rotated.height * scale;
    ctx.drawImage(rotated, 0, 0, scaledCanvas.width, scaledCanvas.height);

    try {
      const result = await worker.recognize(scaledCanvas);
      const text = result.data?.text || '';
      let score = 0;

      for (let keyword of keywords) {
        if (text.includes(keyword)) score++;
      }

      console.log(`🔄 Rotation ${angle}° → Score: ${score}`);
      if (score > bestScore) {
        bestScore = score;
        bestAngle = angle;
      }
    } catch (e) {
      console.warn(`⚠️ OCR failed at ${angle}°`, e);
    }
  }

  await worker.terminate();
  return bestAngle;
}

function preprocessImage(image) {
  const modelSize = 640;
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d', { willReadFrequently: true });

  const imgW = image.width;
  const imgH = image.height;
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

async function handleImageUpload(event) {
  const file = event.target.files[0];
  const img = new Image();

  img.onload = async () => {
    loader.style.display = 'block';

    const rotation = await detectBestRotation(img);
    rotationDisplay.textContent = `📐 Rotation Applied: ${rotation}°`;
    const correctedCanvas = rotateImage(img, rotation);

    canvas.width = correctedCanvas.width;
    canvas.height = correctedCanvas.height;
    ctx.drawImage(correctedCanvas, 0, 0);

    const inputTensor = preprocessImage(correctedCanvas);
    const outputMap = await cropperSession.run({ images: inputTensor });

    const outputTensor = Object.values(outputMap)[0];
    if (!outputTensor || !outputTensor.data || !outputTensor.dims) {
      console.error("❌ Invalid model output format:", outputTensor);
      loader.style.display = 'none';
      return;
    }

    const rawData = outputTensor.data;
    const [batch, channels, numDetections] = outputTensor.dims;

    const boxes = [];
    for (let i = 0; i < numDetections; i++) {
      const x = rawData[i];
      const y = rawData[i + numDetections];
      const w = rawData[i + 2 * numDetections];
      const h = rawData[i + 3 * numDetections];
      const conf = rawData[i + 4 * numDetections];
      if (conf > 0.4) boxes.push({ x, y, w, h, conf });
    }

    const finalBoxes = nonMaxSuppression(boxes);
    const scale = 1 / preprocessImage.lastScale;
    const padX = preprocessImage.lastPadX;
    const padY = preprocessImage.lastPadY;

    const zip = new JSZip();
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

      const cropCanvas = document.createElement('canvas');
      const cropCtx = cropCanvas.getContext('2d');
      cropCanvas.width = w;
      cropCanvas.height = h;
      cropCtx.drawImage(correctedCanvas, left, top, w, h, 0, 0, w, h);

      const blob = await new Promise(resolve => cropCanvas.toBlob(resolve, 'image/png'));
      zip.file(`aadhaar_crop_${i + 1}.png`, blob);
    }

    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const zipLink = document.createElement('a');
    zipLink.href = URL.createObjectURL(zipBlob);
    zipLink.download = 'aadhaar_crops.zip';
    zipLink.click();

    loader.style.display = 'none';
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);