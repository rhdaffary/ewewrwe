let model;
async function loadModel() {
    model = await tf.loadLayersModel('model_web/model.json');
    console.log("Model berhasil dimuat!");
}

async function classifyFrame() {
    if (!model) return;
    const video = document.getElementById("video");
    const tensor = tf.browser.fromPixels(video)
        .resizeNearestNeighbor([64, 64])
        .toFloat()
        .expandDims();
    
    const prediction = model.predict(tensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];

    // Ubah hasil ke teks sesuai label dataset
    const labels = ['A', 'B', 'C', 'D']; // Ganti sesuai datasetmu
    document.getElementById("translation-result").innerText = `Terjemahan: ${labels[predictedClass]}`;

    requestAnimationFrame(classifyFrame);
}

document.addEventListener("DOMContentLoaded", () => {
    loadModel();
});