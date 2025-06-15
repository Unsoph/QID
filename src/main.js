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

  // Set canvas size for preprocessing
  canvas.width = width;
  canvas.height = height;

  // Draw image to canvas
  ctx.drawImage(image, 0, 0, width, height);

  // Get image data
  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;

  // Normalize pixel values and rearrange to CHW
  const input = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    input[i] = data[i * 4] / 255;       // R
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

    // 🔍 Debug: print model output info
    console.log("Model output:", output);
    console.log("Output keys:", Object.keys(output));
    console.log("Output tensor:", output[Object.keys(output)[0]]);
    console.log("Output shape:", output[Object.keys(output)[0]].dims);

    alert("Inference complete! Check the browser console.");
  };

  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
