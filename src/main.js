import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

async function init() {
  session = await ort.InferenceSession.create('./best.onnx');
  console.log("ONNX model loaded.");
}

function rotateIfVertical(image) {
  if (image.naturalHeight > image.naturalWidth) {
    const canvasTemp = document.createElement('canvas');
    const ctxTemp = canvasTemp.getContext('2d');
    canvasTemp.width = image.naturalHeight;
    canvasTemp.height = image.naturalWidth;

    ctxTemp.translate(canvasTemp.width / 2, canvasTemp.height / 2);
    ctxTemp.rotate(-Math.PI / 2);
    ctxTemp.drawImage(image, -image.naturalWidth / 2, -image.naturalHeight / 2);

    const rotatedImage = new Image();
    rotatedImage.src = canvasTemp.toDataURL();
    return rotatedImage;
  }
  return image;
}

function rotate180(image) {
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d');
  canvasTemp.width = image.naturalWidth;
  canvasTemp.height = image.naturalHeight;

  ctxTemp.translate(canvasTemp.width, canvasTemp.height);
  ctxTemp.rotate(Math.PI);
  ctxTemp.drawImage(image, 0, 0);

  const rotatedImage = new Image();
  rotatedImage.src = canvasTemp.toDataURL();
  return rotatedImage;
}

function preprocessImage(image) {
  const modelSize = 640;
  const canvasTemp = document.createElement('canvas');
  const ctxTemp = canvasTemp.getContext('2d', { willReadFrequently: true });

  const imgW = image.naturalWidth;
  const imgH = image.naturalHeight;
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

async function detectWithBestOrientation(image) {
  const rotated = rotateIfVertical(image);
  const candidates = [rotated, rotate180(rotated)];

  let bestBoxes = [];
  let bestImage = null;
  let bestConf = -Infinity;

  for (const testImg of candidates) {
    await new Promise(res => (testImg.onload = res));
    const inputTensor = preprocessImage(testImg);
    const feeds = { images: inputTensor };
    const output = await session.run(feeds);
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
    const topConf = finalBoxes.length > 0 ? finalBoxes[0].conf : 0;

    if (topConf > bestConf) {
      bestConf = topConf;
      bestBoxes = finalBoxes;
      bestImage = testImg;
    }
  }

  return { boxes: bestBoxes, image: bestImage };
}

async function handleImageUpload(event) {
  const file = event.target.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const { boxes, image } = await detectWithBestOrientation(img);

    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);

    const scale = 1 / preprocessImage.lastScale;
    const padX = preprocessImage.lastPadX;
    const padY = preprocessImage.lastPadY;

    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'lime';

    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const x = (box.x - padX) * scale;
      const y = (box.y - padY) * scale;
      const w = box.w * scale;
      const h = box.h * scale;
      const left = x - w / 2;
      const top = y - h / 2;

      ctx.strokeRect(left, top, w, h);
      ctx.fillText(`Conf: ${box.conf.toFixed(2)}`, left, top > 20 ? top - 5 : top + 15);

      cropAndDownload(image, left, top, w, h, i + 1);
    }
  };
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);