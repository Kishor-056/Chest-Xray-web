/**
 * X-Ray Image Validator — TensorFlow.js MobileNet + Sketch/Document Detector
 *
 * Strategy:
 *   1. Basic sanity (size, solid black/white)
 *   2. COLOR DETECTOR (NEW):
 *        Real chest X-rays are grayscale — R, G, B channels are nearly identical.
 *        Color photos (selfies, landscapes, etc.) have high per-pixel channel deviation.
 *        If avgColorDeviation > 15 → reject immediately as a colour image.
 *   3. Sketch / document detector:
 *        Pencil sketches & scanned paper have > 50% near-white pixels (paper background).
 *        X-rays NEVER have more than ~30% near-pure-white pixels.
 *   4. MobileNet classifier:
 *        Real photos  → MobileNet recognises "jersey / person / face..." → HIGH confidence → REJECT
 *        Chest X-rays → MobileNet can't match any ImageNet class          → LOW confidence  → ALLOW
 */

import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

// ── Model cache ───────────────────────────────────────────────────────────────
let _model = null;
let _loadPromise = null;

const loadModel = () => {
    if (_model) return Promise.resolve(_model);
    if (_loadPromise) return _loadPromise;

    _loadPromise = mobilenet.load({ version: 2, alpha: 0.5 })
        .then((m) => { _model = m; return m; })
        .catch((err) => { _loadPromise = null; throw err; });

    return _loadPromise;
};

// ── Keywords that mean "definitely a real-world image, not an X-ray" ─────────
const REAL_WORLD_KEYWORDS = [
    // people & body parts
    'person', 'people', 'man', 'woman', 'boy', 'girl',
    'face', 'head', 'neck', 'hand', 'body',
    // clothing / accessories
    'jersey', 'shirt', 't-shirt', 'sweatshirt', 'uniform', 'sportswear',
    'cap', 'hat', 'helmet', 'glasses', 'suit', 'jacket', 'coat',
    // art & drawings  ← NEW
    'sketch', 'drawing', 'portrait', 'cartoon', 'illustration', 'comic',
    'pencil', 'charcoal', 'artwork', 'caricature', 'doodle', 'painting',
    'sculpture', 'statue',
    // animals
    'dog', 'cat', 'bird', 'horse', 'animal', 'pet',
    // vehicles
    'car', 'truck', 'bus', 'vehicle', 'bike', 'bicycle', 'motorcycle',
    // scenes / objects
    'building', 'house', 'tree', 'flower', 'grass', 'sky', 'road',
    'phone', 'laptop', 'computer', 'keyboard', 'monitor', 'bottle', 'cup',
    'food', 'pizza', 'burger', 'fruit',
    'document', 'book', 'paper', 'text', 'envelope',
];

const isRealWorldLabel = (label) => {
    const lower = label.toLowerCase();
    return REAL_WORLD_KEYWORDS.some((kw) => lower.includes(kw));
};

// ── Pixel analysis helpers ────────────────────────────────────────────────────
const getPixelStats = (file) =>
    new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const SIZE = 128;
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = canvas.height = SIZE;
                ctx.drawImage(img, 0, 0, SIZE, SIZE);
                const d = ctx.getImageData(0, 0, SIZE, SIZE).data;

                let sum = 0;
                let nearWhiteCount = 0;   // brightness > 195
                let nearBlackCount = 0;   // brightness < 30
                let colorDeviationSum = 0; // sum of per-pixel channel deviation from mean
                const total = SIZE * SIZE;

                for (let i = 0; i < d.length; i += 4) {
                    const r = d[i], g = d[i + 1], b = d[i + 2];
                    const bri = (r + g + b) / 3;
                    sum += bri;
                    if (bri > 195) nearWhiteCount++;
                    if (bri < 30) nearBlackCount++;
                    // Per-pixel colorfulness: average absolute deviation of channels from their mean
                    const dev = (Math.abs(r - bri) + Math.abs(g - bri) + Math.abs(b - bri)) / 3;
                    colorDeviationSum += dev;
                }

                resolve({
                    width: img.width,
                    height: img.height,
                    avgBrightness: sum / total,
                    nearWhiteRatio: nearWhiteCount / total,   // fraction of near-white pixels
                    nearBlackRatio: nearBlackCount / total,
                    avgColorDeviation: colorDeviationSum / total, // 0 = pure grayscale, high = colorful
                    imageElement: img,
                });
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });

// ── Main export ───────────────────────────────────────────────────────────────
export const validateImageLocally = async (file) => {

    // ── Step 1: Must be an image file ────────────────────────────────────────
    if (!file.type.startsWith('image/')) {
        const msg = 'File is not an image. Please upload a JPEG or PNG chest X-ray.';
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }

    // ── Step 2: Get pixel statistics ─────────────────────────────────────────
    let stats;
    try {
        stats = await getPixelStats(file);
    } catch {
        const msg = 'Could not read the image file.';
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }

    const { width, height, avgBrightness, nearWhiteRatio, nearBlackRatio, avgColorDeviation, imageElement } = stats;

    // ── Step 3: Minimum size ──────────────────────────────────────────────────
    if (width < 80 || height < 80) {
        const msg = `Image too small (${width}×${height}px). Medical X-rays are at least 200×200 px.`;
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }

    // ── Step 4: Solid black / solid white ────────────────────────────────────
    if (avgBrightness < 8) {
        const msg = 'Image is almost completely black — not a valid chest X-ray.';
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }
    if (avgBrightness > 247) {
        const msg = 'Image is almost completely white — not a valid chest X-ray.';
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }

    // ── Step 5: COLOR IMAGE DETECTOR ──────────────────────────────────────────
    //
    //  Chest X-rays are grayscale: R ≈ G ≈ B for every pixel.
    //  avgColorDeviation measures average per-pixel spread between R, G, B channels.
    //  Pure grayscale images → avgColorDeviation ≈ 0–8.
    //  Natural colour photos → avgColorDeviation typically 15–60+.
    //  Threshold of 15 safely catches all colour photos while allowing slightly
    //  tinted or JPEG-compressed grayscale X-rays.
    //
    if (avgColorDeviation > 15) {
        const msg =
            `❌ This appears to be a colour image (color deviation: ${avgColorDeviation.toFixed(1)}). ` +
            `Chest X-rays are always grayscale. Please upload a valid medical X-ray image.`;
        return { is_valid: false, confidence: 0, reasons: [msg], message: msg };
    }

    // ── Step 6: SKETCH / DOCUMENT / DIAGRAM DETECTOR ─────────────────────────
    //
    //  TWO sub-checks:
    //
    //  A) Near-white ratio:
    //     Pencil sketches & flowcharts sit on white paper → > 45% near-white pixels.
    //     Chest X-rays are never mostly white (dark lungs dominate).
    //
    //  B) Bimodal high-contrast check:
    //     Flowcharts / diagrams / text documents = sharp black ink on white paper.
    //     → nearBlack + nearWhite account for > 75% of pixels (almost no mid-gray).
    //     Chest X-rays have rich grayscale gradients → mid-gray pixels > 35%.
    //
    const midGrayRatio = 1 - nearWhiteRatio - nearBlackRatio;

    if (nearWhiteRatio > 0.45) {
        const msg =
            `This image appears to be a sketch, drawing, or document ` +
            `(${(nearWhiteRatio * 100).toFixed(0)}% near-white pixels — looks like paper). ` +
            `Chest X-rays are not paper-white. Please upload a real medical X-ray image.`;
        return { is_valid: false, confidence: nearWhiteRatio, reasons: [msg], message: msg };
    }

    if ((nearBlackRatio + nearWhiteRatio) > 0.75 && midGrayRatio < 0.25) {
        const msg =
            `This image is a high-contrast black-on-white document or diagram ` +
            `(${((nearBlackRatio + nearWhiteRatio) * 100).toFixed(0)}% pixels are near-black or near-white). ` +
            `Chest X-rays have rich gray gradients. Please upload a real medical X-ray image.`;
        return { is_valid: false, confidence: nearBlackRatio + nearWhiteRatio, reasons: [msg], message: msg };
    }


    // ── Step 7: Load MobileNet (cached after first download ~5 MB) ───────────
    let model;
    try {
        model = await loadModel();
    } catch (err) {
        console.warn('MobileNet unavailable — basic checks passed:', err);
        return {
            is_valid: true,
            confidence: 0.5,
            reasons: [],
            message: 'AI model unavailable — basic checks passed. Backend will validate further.',
        };
    }

    // ── Step 8: MobileNet classification ─────────────────────────────────────
    let predictions;
    try {
        predictions = await model.classify(imageElement, 5);
    } catch (err) {
        console.warn('MobileNet classify error:', err);
        return { is_valid: true, confidence: 0.5, reasons: [], message: 'Classification failed — basic checks passed.' };
    }

    console.debug('[X-ray Validator] MobileNet predictions:', predictions);

    const top = predictions[0];
    const topP = top?.probability ?? 0;
    const topLabel = top?.className ?? '';

    // Rule A: Recognised real-world label with meaningful confidence → reject
    // Threshold lowered to 0.10 so that artistic/sketch renderings are also caught
    if (topP > 0.10 && isRealWorldLabel(topLabel)) {
        const readableLabel = topLabel.split(',')[0].trim();
        const msg = `This image looks like "${readableLabel}", not a chest X-ray. Please upload a valid medical X-ray image.`;
        return { is_valid: false, confidence: topP, reasons: [msg], message: msg };
    }

    // Rule B: Any class confidently predicted → not an X-ray
    if (topP > 0.40) {
        const readableLabel = topLabel.split(',')[0].trim();
        const msg = `Image appears to be "${readableLabel}" — not a chest X-ray. Please upload a valid medical X-ray image.`;
        return { is_valid: false, confidence: topP, reasons: [msg], message: msg };
    }

    // Rule C: Low top-confidence = MobileNet couldn't recognise it = likely medical image
    return {
        is_valid: true,
        confidence: 1 - topP,
        reasons: [],
        message: 'Image appears to be a valid chest X-ray.',
    };
};
