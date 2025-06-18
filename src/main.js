import * as ort from 'onnxruntime-web';
import * as EXIF from 'exif-js';

// Load YOLOv8 ONNX model
const modelPath = 'model.onnx'; // adjust if in a subfolder

let session;

async function loadModel() {
  session = await ort.InferenceSession.create(modelPath);
}
loadModel();

document.getElementById('image-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    EXIF.getData(file, function () {
      const orientation = EXIF.getTag(this, 'Orientation');
      fixOrientation(img, orientation, (orientedImg) => {
        detectAndCrop(orientedImg);
      });
    });
  };
  img.src = URL.createObjectURL(file);
});

function fixOrientation(img, orientation, callback) {
  if (!orientation || orientation === 1) {
    callback(img);
    return;
  }

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const width = img.width;
  const height = img.height;

  if (orientation > 4) {
    canvas.width = height;
    canvas.height = width;
  } else {
    canvas.width = width;
    canvas.height = height;
  }

  switch (orientation) {
    case 2: ctx.transform(-1, 0, 0, 1, width, 0); break;
    case 3: ctx.transform(-1, 0, 0, -1, width, height); break;
    case 4: ctx.transform(1, 0, 0, -1, 0, height); break;
    case 5: ctx.transform(0, 1, 1, 0, 0, 0); break;
    case 6: ctx.transform(0, 1, -1, 0, height, 0); break;
    case 7: ctx.transform(0, -1, -1, 0, height, width); break;
    case 8: ctx.transform(0, -1, 1, 0, 0, width); break;
    default: break;
  }

  ctx.drawImage(img, 0, 0);

  const newImg = new Image();
  newImg.onload = () => callback(newImg);
  newImg.src = canvas.toDataURL();
}

async function detectAndCrop(image) {
  const canvas = document.getElementById('output-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);

  // Preprocess image into tensor
  const inputTensor = await preprocess(image);

  const feeds = { images: inputTensor };
  const results = await session.run(feeds);

  const output = results[Object.keys(results)[0]].data;

  // Parse YOLOv8 output
  const boxes = parseDetections(output, canvas.width, canvas.height);

  boxes.forEach((box, i) => {
    const [x1, y1, x2, y2, conf] = box;

    // Draw box
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Crop and download
    cropAndDownload(image, box, i);
  });
}

async function preprocess(image) {
  const modelWidth = 640;
  const modelHeight = 640;

  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = modelWidth;
  offscreenCanvas.height = modelHeight;
  const ctx = offscreenCanvas.getContext('2d');
  ctx.drawImage(image, 0, 0, modelWidth, modelHeight);

  const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
  const { data } = imageData;

  const float32Data = new Float32Array(modelWidth * modelHeight * 3);
  for (let i = 0; i < modelWidth * modelHeight; i++) {
    float32Data[i] = data[i * 4] / 255;       // R
    float32Data[i + modelWidth * modelHeight] = data[i * 4 + 1] / 255; // G
    float32Data[i + 2 * modelWidth * modelHeight] = data[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
}

function parseDetections(output, width, height, confThreshold = 0.25) {
  const boxes = [];

  const numDetections = output.length / 6; // YOLOv8 gives [x1, y1, x2, y2, conf, class]
  for (let i = 0; i < numDetections; i++) {
    const x1 = output[i * 6];
    const y1 = output[i * 6 + 1];
    const x2 = output[i * 6 + 2];
    const y2 = output[i * 6 + 3];
    const conf = output[i * 6 + 4];
    const cls = output[i * 6 + 5];

    if (conf > confThreshold) {
      // Rescale from model (640x640) to original image size
      boxes.push([x1 * width / 640, y1 * height / 640, x2 * width / 640, y2 * height / 640, conf, cls]);
    }
  }

  return boxes;
}

function cropAndDownload(image, box, index) {
  const [x1, y1, x2, y2] = box;

  const cropWidth = x2 - x1;
  const cropHeight = y2 - y1;

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropWidth;
  cropCanvas.height = cropHeight;

  const ctx = cropCanvas.getContext('2d');
  ctx.drawImage(image, x1, y1, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

  cropCanvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `aadhaar_crop_${index + 1}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, 'image/jpeg');
}