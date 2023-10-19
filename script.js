// Ambil data dari Google Drive
const url =
  "https://drive.google.com/uc?id=1g6EQGwMZvstP7Q5uioca4Yk30n6DlfgU&export=download&confirm=t";
const localZip = "house_image.zip";

// Fungsi untuk membuat direktori jika tidak ada
function createDirIfNotExists(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
    console.log(`Created directory: ${dir}`);
  }
}

// Fungsi untuk mengekstrak berkas ZIP
function extractZip(zipPath, targetDir) {
  const AdmZip = require("adm-zip");
  const zip = new AdmZip(zipPath);
  zip.extractAllTo(targetDir, true);
  console.log(`Extracted ZIP file: ${zipPath}`);
}

// Fungsi untuk membagi dataset menjadi pelatihan dan validasi
function splitDataset(sourceDir, trainDir, testDir, splitSize) {
  const fs = require("fs");
  const path = require("path");
  const files = fs.readdirSync(sourceDir);
  files.forEach((file) => {
    const sourcePath = path.join(sourceDir, file);
    const destDir = Math.random() < splitSize ? trainDir : testDir;
    const destPath = path.join(destDir, file);
    fs.copyFileSync(sourcePath, destPath);
  });
}

// Fungsi untuk memuat gambar dari input file
async function loadImage(inputElement) {
  const file = inputElement.files[0];
  const reader = new FileReader();

  return new Promise((resolve, reject) => {
    reader.onload = function (e) {
      const image = new Image();
      image.src = e.target.result;
      image.onload = () => resolve(tf.browser.fromPixels(image));
      image.onerror = (error) => reject(error);
    };

    reader.readAsDataURL(file);
  });
}

async function loadCustomBaseModel() {
  const imageShape = [224, 224, 3]; // Sesuaikan dengan bentuk gambar yang Anda gunakan

  const customBaseModel = await tf.loadLayersModel(
    "path/to/your/custom/model/model.json"
  );
  customBaseModel.trainable = true;

  const flattenLayer = tf.layers.flatten();
  const denseLayer1 = tf.layers.dense({ units: 256, activation: "relu" });
  const batchNorm1 = tf.layers.batchNormalization();
  const denseLayer2 = tf.layers.dense({ units: 128, activation: "relu" });
  const batchNorm2 = tf.layers.batchNormalization();
  const denseLayer3 = tf.layers.dense({
    units: 64,
    activation: "relu",
    kernelRegularizer: tf.regularizers.l2(),
    activityRegularizer: tf.regularizers.l1(),
    biasRegularizer: tf.regularizers.l1(),
  });
  const dropoutLayer = tf.layers.dropout({ rate: 0.5 });
  const outputLayer = tf.layers.dense({ units: 1, activation: "linear" });

  const input = tf.input({ shape: imageShape });
  let x = customBaseModel.apply(input);
  x = flattenLayer.apply(x);
  x = denseLayer1.apply(x);
  x = batchNorm1.apply(x);
  x = denseLayer2.apply(x);
  x = batchNorm2.apply(x);
  x = denseLayer3.apply(x);
  x = dropoutLayer.apply(x);
  const output = outputLayer.apply(x);

  const model = tf.model({ inputs: input, outputs: output });

  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

let model; // Simpan model di sini
// Inisialisasi model saat dokumen dimuat
document.addEventListener("DOMContentLoaded", async function () {
  model = await loadCustomBaseModel();
});

//ProcessImage
async function processImage(inputElement, splitSize) {
  // Persiapkan direktori dan dataset
  createDirIfNotExists("/content");
  createDirIfNotExists("/content/Houses-Images");
  createDirIfNotExists("/content/Houses-Images/not_eligible");
  createDirIfNotExists("/content/Houses-Images/eligible");
  const trainDir = "/content/Houses-Images/train";
  const testDir = "/content/Houses-Images/test";
  createDirIfNotExists(trainDir);
  createDirIfNotExists(testDir);

  // Ekstrak berkas ZIP
  extractZip(localZip, "/content");

  // Bagi dataset
  splitDataset(
    "/content/Houses-Images/not_eligible",
    `${trainDir}/not_eligible`,
    `${testDir}/not_eligible`,
    splitSize
  );
  splitDataset(
    "/content/Houses-Images/eligible",
    `${trainDir}/eligible`,
    `${testDir}/eligible`,
    splitSize
  );

  // Mulai memuat gambar
  const imageTensor = await loadImage(inputElement);

  // Lakukan normalisasi
  const normalizedImage = tf
    .div(imageTensor, 255.0)
    .reshape([1, ...image_shape]);

  // Lakukan prediksi menggunakan model
  const prediction = model.predict(normalizedImage);

  // Proses hasil prediksi di sini
  const resultElement = document.getElementById("predictionResult");
  resultElement.innerText = `Hasil Prediksi: ${prediction.dataSync()}`;
}

async function startPrediction() {
  const inputElement = document.getElementById("fileInput");
  const splitSize = 0.8; // Sesuaikan dengan ukuran split yang diinginkan
  await processImage(inputElement, splitSize);
}
