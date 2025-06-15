import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

async function init() {
  session = await ort.InferenceSession.create('best.onnx'); // ✅ FIXED PATH
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
    console.log("Model output:", output);
    console.log("Output keys:", Object.keys(output));
    console.log("Output tensor:", output[Object.keys(output)[0]]);
    console.log("Output shape:", output[Object.keys(output)[0]].dims);

    // 🔍 Extract and draw boxes
    const outputData = output[Object.keys(output)[0]].data;
    const outputDims = output[Object.keys(output)[0]].dims;

    const boxes = [];
    const [batch, channels, numPreds] = outputDims;

    for (let i = 0; i < numPreds; i++) {
      const x = outputData[i * 5];
      const y = outputData[i * 5 + 1];
      const w = outputData[i * 5 + 2];
      const h = outputData[i * 5 + 3];
      const conf = outputData[i * 5 + 4];

      const confThreshold = 0.8;
      if (conf > confThreshold) {
        boxes.push({ x, y, w, h, conf });
      }
    }

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;

    for (const box of boxes) {
      const left = box.x - box.w / 2;
      const top = box.y - box.h / 2;
      ctx.strokeRect(left, top, box.w, box.h);
    }

    alert(`Inference complete! ${boxes.length} object(s) detected.`);
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
