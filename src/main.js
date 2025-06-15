import * as ort from 'onnxruntime-web';

const imageInput = document.getElementById('image-input');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let session;

async function init() {
  session = await ort.InferenceSession.create('best.onnx');
  console.log('✅ ONNX model loaded.');
}

function preprocessImage(image) {
  const MODEL_SIZE = 640;
  const scale = Math.min(MODEL_SIZE / image.width, MODEL_SIZE / image.height);
  const newW = Math.round(image.width * scale);
  const newH = Math.round(image.height * scale);
  const padX = (MODEL_SIZE - newW) / 2;
  const padY = (MODEL_SIZE - newH) / 2;

  const tmp = document.createElement('canvas');
  const tctx = tmp.getContext('2d');
  tmp.width = tmp.height = MODEL_SIZE;

  tctx.fillStyle = 'rgb(114,114,114)';
  tctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  tctx.drawImage(image, padX, padY, newW, newH);

  const imgData = tctx.getImageData(0,0,MODEL_SIZE,MODEL_SIZE).data;
  const input = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const area = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < area; i++) {
    input[i] = imgData[i*4]/255;
    input[i+area] = imgData[i*4+1]/255;
    input[i+2*area] = imgData[i*4+2]/255;
  }

  return { tensor: new ort.Tensor('float32', input, [1,3,MODEL_SIZE,MODEL_SIZE]), scale, padX, padY };
}

function calculateIoU(a, b) {
  const x1 = Math.max(a.x - a.w/2, b.x - b.w/2);
  const y1 = Math.max(a.y - a.h/2, b.y - b.h/2);
  const x2 = Math.min(a.x + a.w/2, b.x + b.w/2);
  const y2 = Math.min(a.y + a.h/2, b.y + b.h/2);
  const inter = Math.max(0, x2-x1)*Math.max(0, y2-y1);
  return inter / (a.w*a.h + b.w*b.h - inter);
}

function nonMaxSuppression(boxes, iouThreshold=0.5) {
  boxes.sort((a,b)=>b.conf - a.conf);
  const picked = [];
  for (let box of boxes) {
    if (!picked.some(pb => calculateIoU(box, pb) > iouThreshold)) {
      picked.push(box);
    }
  }
  return picked;
}

async function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file || !session) return alert('No image or model loaded.');

  const img = new Image();
  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img,0,0);

    const {tensor, scale, padX, padY} = preprocessImage(img);
    const output = await session.run({ images: tensor });
    const outputName = Object.keys(output)[0];
    const raw = output[outputName].data;
    const dims = output[outputName].dims;  // [1,5,8400]
    const area = 8400;

    // Heatmap overlay
    const ratioW = img.width / 640, ratioH = img.height / 640;
    for (let i = 0; i < area; i++) {
      const x = raw[i], y = raw[area+i], w = raw[2*area+i], h = raw[3*area+i], conf = raw[4*area+i];
      const left = (x - w/2)*ratioW;
      const top = (y - h/2)*ratioH;
      const bw = w*ratioW;
      const bh = h*ratioH;
      ctx.beginPath();
      ctx.rect(left,top,bw,bh);
      ctx.strokeStyle = `rgba(0,255,0,${conf.toFixed(2)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = `rgba(255,0,0,${(1-conf).toFixed(2)})`;
      ctx.fillRect(left,top,bw,bh);
    }

    // Parse and filter boxes (post heatmap)
    const boxes = [];
    for (let i = 0; i < area; i++) {
      const conf = raw[4*area+i];
      if (conf > 0.3) {
        const x = raw[i], y=raw[area+i], w=raw[2*area+i], h=raw[3*area+i];
        boxes.push({ x, y, w, h, conf });
      }
    }
    const final = nonMaxSuppression(boxes);
    console.log(`Raw: ${boxes.length}, Filtered: ${final.length}`);

    // Draw final boxes
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'lime';
    final.forEach(box=> {
      const left = (box.x - box.w/2)*ratioW;
      const top = (box.y - box.h/2)*ratioH;
      const bw = box.w*ratioW, bh=box.h*ratioH;
      ctx.strokeRect(left,top,bw,bh);
      ctx.fillText(box.conf.toFixed(2), left, top-5);
    });

    alert(`Finished: ${final.length} detected`);
  };
  img.src = URL.createObjectURL(file);
}

document.addEventListener('DOMContentLoaded', init);
imageInput.addEventListener('change', handleImageUpload);
