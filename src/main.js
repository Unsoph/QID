import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('imageInput');
const resultCanvas = document.getElementById('resultCanvas');
const ctx = resultCanvas.getContext('2d');

imageInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Resize image to expected YOLOv8 input size (640x640)
    const resized = document.createElement('canvas');
    resized.width = 640;
    resized.height = 640;
    resized.getContext('2d').drawImage(img, 0, 0, 640, 640);

    const imageData = resized.getContext('2d').getImageData(0, 0, 640, 640);
    const data = imageData.data;
    
    // Convert RGBA to Float32 normalized tensor [1, 3, 640, 640]
    const input = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
      input[i] = data[i * 4] / 255.0; // R
      input[i + 640 * 640] = data[i * 4 + 1] / 255.0; // G
      input[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);

    const session = await ort.InferenceSession.create('./best.onnx');
    const feeds = { images: tensor };

    const results = await session.run(feeds);
    console.log('Output:', results);
  };
});
