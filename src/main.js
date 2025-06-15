import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
let session;

async function init() {
  session = await ort.InferenceSession.create('./public/best.onnx');
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
    // Resize canvas to image dimensions for final output
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

      // Green stroke for confidence
      ctx.beginPath();
      ctx.rect(left, top, boxWidth, boxHeight);
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(0,255,0,${conf.toFixed(2)})`;
      ctx.stroke();

      // Red fill for low confidence
      ctx.fillStyle = `rgba(255,0,0,${(1 - conf).toFixed(2)})`;
      ctx.fillRect(left, top, boxWidth, boxHeight);
    }

    alert("Heatmap drawn. Check bounding boxes and color intensity.");
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
