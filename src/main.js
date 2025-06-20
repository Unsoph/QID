import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

// Aadhaar keywords for orientation detection
const AADHAAR_KEYWORDS = [
    "government of india", "govt of india", "dob", "year of birth",
    "male", "female", "vid", "aadhaar", "आधार", "भारत सरकार"
];

async function init() {
  session = await ort.InferenceSession.create('./best.onnx');
  console.log("ONNX model loaded.");
}

// Rotate image by specified angle
function rotateImage(image, angle) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  if (angle === 0) {
    canvas.width = image.width || image.naturalWidth;
    canvas.height = image.height || image.naturalHeight;
    ctx.drawImage(image, 0, 0);
    return canvas;
  }
  
  const w = image.width || image.naturalWidth;
  const h = image.height || image.naturalHeight;
  
  if (angle === 90 || angle === 270) {
    canvas.width = h;
    canvas.height = w;
  } else {
    canvas.width = w;
    canvas.height = h;
  }
  
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate((angle * Math.PI) / 180);
  ctx.drawImage(image, -w / 2, -h / 2);
  
  return canvas;
}

// Enhanced OCR using browser's text detection (if available)
async function extractTextFromImage(canvas) {
  try {
    // Try to use browser's native text detection if available
    if ('DetectText' in window) {
      const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
      const results = await window.DetectText(imageData);
      return results.map(r => r.text.toLowerCase()).join(' ');
    }
    
    // Fallback: Use Tesseract.js if available
    if (typeof Tesseract !== 'undefined') {
      const { data: { text } } = await Tesseract.recognize(canvas, 'eng+hin');
      return text.toLowerCase();
    }
    
    // If no OCR available, return empty string
    console.warn('No OCR capability available. Skipping text-based orientation detection.');
    return '';
  } catch (error) {
    console.warn('OCR failed:', error);
    return '';
  }
}

// Calculate image sharpness/clarity score
function calculateSharpnessScore(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // Convert to grayscale and calculate Laplacian variance (measure of sharpness)
  const gray = [];
  for (let i = 0; i < data.length; i += 4) {
    gray.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  
  let laplacianSum = 0;
  const width = canvas.width;
  const height = canvas.height;
  
  // Apply Laplacian kernel
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const laplacian = 
        -gray[idx - width - 1] - gray[idx - width] - gray[idx - width + 1] +
        -gray[idx - 1] + 8 * gray[idx] - gray[idx + 1] +
        -gray[idx + width - 1] - gray[idx + width] - gray[idx + width + 1];
      laplacianSum += laplacian * laplacian;
    }
  }
  
  return laplacianSum / ((width - 2) * (height - 2));
}

// Detect text orientation using edge detection
function detectTextOrientation(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const width = canvas.width;
  const height = canvas.height;
  
  // Convert to grayscale
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < data.length; i += 4) {
    gray[i / 4] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  
  // Apply Sobel edge detection
  const sobelX = [];
  const sobelY = [];
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      
      // Sobel X (vertical edges)
      const gx = 
        -gray[idx - width - 1] + gray[idx - width + 1] +
        -2 * gray[idx - 1] + 2 * gray[idx + 1] +
        -gray[idx + width - 1] + gray[idx + width + 1];
      
      // Sobel Y (horizontal edges)
      const gy = 
        -gray[idx - width - 1] - 2 * gray[idx - width] - gray[idx - width + 1] +
        gray[idx + width - 1] + 2 * gray[idx + width] + gray[idx + width + 1];
      
      sobelX.push(gx);
      sobelY.push(gy);
    }
  }
  
  // Calculate dominant edge direction
  let horizontalEdges = 0;
  let verticalEdges = 0;
  
  for (let i = 0; i < sobelX.length; i++) {
    const magnitude = Math.sqrt(sobelX[i] * sobelX[i] + sobelY[i] * sobelY[i]);
    if (magnitude > 50) { // Threshold for significant edges
      if (Math.abs(sobelX[i]) > Math.abs(sobelY[i])) {
        verticalEdges++;
      } else {
        horizontalEdges++;
      }
    }
  }
  
  return { horizontalEdges, verticalEdges };
}

// Find the best orientation for the image
async function findBestOrientation(image) {
  console.log("🔄 Checking image orientation...");
  
  const rotations = [0, 90, 180, 270];
  const scores = [];
  
  for (const angle of rotations) {
    const rotatedCanvas = rotateImage(image, angle);
    
    // Calculate sharpness score
    const sharpnessScore = calculateSharpnessScore(rotatedCanvas);
    
    // Get edge direction information
    const edgeInfo = detectTextOrientation(rotatedCanvas);
    const edgeScore = edgeInfo.horizontalEdges > edgeInfo.verticalEdges ? 100 : 0;
    
    // Try to extract text for keyword matching
    const extractedText = await extractTextFromImage(rotatedCanvas);
    const keywordScore = AADHAAR_KEYWORDS.reduce((score, keyword) => {
      return extractedText.includes(keyword) ? score + 200 : score;
    }, 0);
    
    const totalScore = sharpnessScore + edgeScore + keywordScore;
    
    scores.push({
      angle,
      sharpnessScore: sharpnessScore.toFixed(1),
      edgeScore,
      keywordScore,
      totalScore: totalScore.toFixed(1)
    });
    
    console.log(`Rotation ${angle}° → Sharpness: ${sharpnessScore.toFixed(1)}, Edge: ${edgeScore}, Keywords: ${keywordScore}, Total: ${totalScore.toFixed(1)}`);
  }
  
  // Find the best rotation
  const bestRotation = scores.reduce((best, current) => 
    parseFloat(current.totalScore) > parseFloat(best.totalScore) ? current : best
  );
  
  console.log(`✅ Best orientation: ${bestRotation.angle}°`);
  return bestRotation.angle;
}

// Apply orientation correction to image
async function correctOrientation(image) {
  const bestAngle = await findBestOrientation(image);
  
  if (bestAngle === 0) {
    console.log("📍 Image is already in correct orientation");
    return image;
  }
  
  console.log(`🔄 Rotating image by ${bestAngle}°`);
  const correctedCanvas = rotateImage(image, bestAngle);
  
  // Convert canvas back to image
  return new Promise((resolve) => {
    const correctedImage = new Image();
    correctedImage.onload = () => resolve(correctedImage);
    correctedImage.src = correctedCanvas.toDataURL();
  });
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
    try {
      console.log("📸 Image loaded, starting orientation correction...");
      
      // Step 1: Correct image orientation
      const correctedImg = await correctOrientation(img);
      
      // Step 2: Set up canvas with corrected image
      canvas.width = correctedImg.naturalWidth || correctedImg.width;
      canvas.height = correctedImg.naturalHeight || correctedImg.height;
      ctx.drawImage(correctedImg, 0, 0);

      console.log("🤖 Running ML model for Aadhaar detection...");
      
      // Step 3: Process with ML model
      const inputTensor = preprocessImage(correctedImg);
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
      console.log(`🎯 Found ${finalBoxes.length} Aadhaar card(s)`);

      // Step 4: Draw results
      ctx.drawImage(correctedImg, 0, 0, correctedImg.naturalWidth || correctedImg.width, correctedImg.naturalHeight || correctedImg.height);

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

        cropAndDownload(correctedImg, left, top, w, h, i + 1);
      }
      
      console.log("✅ Processing complete!");
      
    } catch (error) {
      console.error("❌ Error processing image:", error);
    }
  };
  
  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);