import * as ort from 'onnxruntime-web';
import JSZip from 'jszip';
import Tesseract from 'tesseract.js';

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

const AADHAAR_KEYWORDS = [
  "government of india", "govt of india", "dob", "year of birth",
  "male", "female", "vid", "aadhaar", "आधार", "भारत सरकार"
];

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

async function getBestRotation(image) {
  let bestRotation = 0;
  let bestScore = -Infinity;

  for (let angle of [0, 90, 180, 270]) {
    const rotatedCanvas = rotateImage(image, angle);
    const dataURL = rotatedCanvas.toDataURL('image/jpeg');
    const { data: { text, words } } = await Tesseract.recognize(
      dataURL,
      'eng',
      { logger: m => console.log(m) }
    );

    const wordConfs = words.map(w => parseFloat(w.conf)).filter(c => !isNaN(c));
    const ocrScore = wordConfs.reduce((a, b) => a + b, 0);
    const textLower = text.toLowerCase();
    const keywordHits = AADHAAR_KEYWORDS.reduce((sum, kw) => sum + textLower.includes(kw), 0);
    const totalScore = ocrScore + keywordHits * 100;

    if (totalScore > bestScore) {
      bestScore = totalScore;
      bestRotation = angle;
    }
  }

  return bestRotation;
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

async function handleImageUpload(event) {
  const file = event.target.files[0];
  const img = new Image();

  img.onload = async () => {
    loader.style.display = 'block';

    const bestRotation = await getBestRotation(img);
    rotationDisplay.textContent = `🖐 Rotation Applied: ${bestRotation}°`;

    const correctedCanvas = rotateImage(img, bestRotation);
    canvas.width = correctedCanvas.width;
    canvas.height = correctedCanvas.height;
    ctx.drawImage(correctedCanvas, 0, 0);

    const inputTensor = preprocessImage(correctedCanvas);
    const output = await cropperSession.run({ images: inputTensor });
    const rawData = output[Object.keys(output)[0]].data;
    const [batch, channels, numDetections] = output[Object.keys(output)[0]].dims;

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