import * as ort from 'onnxruntime-web';

async function runModel() {
  const session = await ort.InferenceSession.create('/best.onnx');
  console.log('✅ Model loaded');

  // Log model input/output names for now
  console.log('Inputs:', session.inputNames);
  console.log('Outputs:', session.outputNames);
}

runModel();

