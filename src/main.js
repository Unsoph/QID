import * as ort from 'onnxruntime-web';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const upload = document.getElementById('upload');

upload.onchange = async (e) => {
  const file = e.target.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    // Resize canvas to match image
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Preprocess image to tensor
    const tensor = await imageToTensor(img);

    // Load model
    const session = await ort.InferenceSession.create('/best.onnx');

    // Inference (check input name using session.inputNames)
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);

    console.log("Inference output:", results);

    // Postprocessing — visualize (you can update this later)
    alert("Inference completed. Check console!");
  };
};

async function imageToTensor(img) {
  const w = img.width;
  const h = img.height;

  // Draw to a temp canvas to get image data
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tmpCtx = tmp.getContext('2d');
  tmpCtx.drawImage(img, 0, 0, w, h);

  const { data } = tmpCtx.getImageData(0, 0, w, h);
  const floatData = new Float32Array(w * h * 3);

  for (let i = 0; i < w * h; i++) {
    floatData[i] = data[i * 4] / 255;           // R
    floatData[i + w * h] = data[i * 4 + 1] / 255; // G
    floatData[i + 2 * w * h] = data[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor('float32', floatData, [1, 3, h, w]);
}
