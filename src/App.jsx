import React, { useState, useRef, useEffect, useCallback } from "react";
import {
    Upload,
    FileImage,
    X,
    CheckCircle,
    Loader,
    AlertCircle,
    Download,
} from "lucide-react";
import ImageZoom from "./components/ImageZoom";

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [resultImageUrl, setResultImageUrl] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const DEFAULT_MODEL = "unet";
    const [detectedSource, setDetectedSource] = useState(null); // 'sar' | 'aerial' | null
    const [detectionConfidence, setDetectionConfidence] = useState(null);
    const [zoomedImage, setZoomedImage] = useState(null); // URL of image to zoom

    // Color mapping for segmentation classes (RGB values)
    const COLOR_MAP = [
        [0, 0, 0],      // Class 0: Background - Black
        [0, 255, 255],  // Class 1: Sheen - Cyan
        [255, 0, 0],    // Class 2: Oil Spill - Red
        [153, 76, 0],   // Class 3: Ship - Brown
        [0, 153, 0],    // Class 4: Land/Vegetation - Green
    ];

    const CLASS_LABELS = [
        "Background/Water",
        "Sheen",
        "Oil Spill",
        "Ship",
        "Land/Vegetation",
    ];

    // Refs for aerial overlay rendering (kept in case we want client-side later)
    const overlayContainerRef = useRef(null);
    const previewImgRef = useRef(null);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("image/")) {
            setSelectedFile(file);
            // Revoke previous preview URL to avoid leaking blob URLs
            if (previewUrl) {
                try {
                    URL.revokeObjectURL(previewUrl);
                } catch (e) {
                    // ignore
                }
            }
            setPreviewUrl(URL.createObjectURL(file));
            setResultImageUrl(null);
            setError(null);
            // reset auto-detection state
            setDetectedSource(null);
            setDetectionConfidence(null);
            // Kick off auto-detection
            runAutoDetect(file);
        } else {
            setError("Please select a valid image file");
        }
    };

    // Try to detect whether the image is SAR or Aerial using a backend endpoint
    const runAutoDetect = async (file) => {
        try {
            const formData = new FormData();
            formData.append("file", file);

            // Attempt to call backend detect endpoint. If it doesn't exist, we'll silently skip.
            const resp = await fetch(
                `http://localhost:8000/detect/sarvsdrone`,
                {
                    method: "POST",
                    body: formData,
                },
            );

            if (!resp.ok) {
                // Not available or error - leave detection as null
                console.warn(
                    "Auto-detect endpoint not available or returned error",
                );
                return;
            }

            const json = await resp.json();
            // Expecting shape: { source: 'sar'|'aerial', confidence: 0.0-1.0 }
            if (json && (json.source === "sar" || json.source === "aerial")) {
                setDetectedSource(json.source);
                    setDetectionConfidence(
                        typeof json.confidence === "number"
                            ? json.confidence
                            : null,
                    );
                // If SAR -> we want to call 'both' endpoint; if aerial -> 'aerial'
                // We don't auto-start prediction here; user still clicks 'Detect Oil Spills'
            }
        } catch (err) {
            console.warn("Auto-detect failed:", err);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            handleFileSelect(files[0]);
        }
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    };

    const processImage = async () => {
        if (!selectedFile) return;

        // Block aerial images
        if (detectedSource === "aerial") {
            setError("Aerial images are not supported. Please upload a SAR (Synthetic Aperture Radar) image.");
            return;
        }

        setIsProcessing(true);
        setError(null);
        setResultImageUrl(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            // Only SAR images are supported - always route to SAR endpoints
            // Always use both models for SAR processing
            const endpoint = "both";

            const response = await fetch(
                `http://localhost:8000/predict/${endpoint}`,
                {
                    method: "POST",
                    body: formData,
                },
            );

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to process image: ${errorText}`);
            }

            // All endpoints now return images
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            // Revoke previous result URL if any
            if (resultImageUrl) {
                try {
                    URL.revokeObjectURL(resultImageUrl);
                } catch (e) {
                    // ignore
                }
            }
            setResultImageUrl(imageUrl);
        } catch (err) {
            setError(`Failed to process image: ${err.message}`);
            console.error("Processing error:", err);
        } finally {
            setIsProcessing(false);
        }
    };

    const downloadResult = () => {
        if (resultImageUrl) {
            const link = document.createElement("a");
            link.href = resultImageUrl;
            const modelLabel = getResultModelLabel();
            const safeLabel = modelLabel.replace(/[^a-z0-9_-]/gi, "_");
            link.download = `${safeLabel}_prediction.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    // Compute a friendly label for the model(s) that produced the current result
    const getResultModelLabel = () => {
        // Only SAR images supported - always use both models
        return "UNet + DeepLabV3+";
    };

    const resetUpload = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResultImageUrl(null);
        setError(null);
        // Clean up object URLs to prevent memory leaks
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        if (resultImageUrl) URL.revokeObjectURL(resultImageUrl);
    };
    

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
            {/* Header */}
            <header className="bg-white/80 backdrop-blur-sm border-b border-blue-100 sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-blue-600 rounded-lg">
                            <FileImage className="h-8 w-8 text-white" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">
                                Oil Spill Detection
                            </h1>
                            <p className="text-gray-600">
                                AI-powered marine pollution analysis
                            </p>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Upload Section */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8">
                            <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
                                <Upload className="h-6 w-6 mr-3 text-blue-600" />
                                Upload SAR Image
                            </h2>

                            {/* Model Selection (automatic) */}
                            <div className="mb-6">
                                <label className="block text-sm font-medium text-gray-700 mb-3">
                                    Image Type
                                </label>
                                <div className="rounded-lg p-4 bg-gray-50 border border-gray-200 text-sm text-gray-700">
                                    Only SAR (Synthetic Aperture Radar) images are supported.
                                    SAR images will run both UNet and DeepLabV3+ segmentation models.
                                </div>
                            </div>

                            {/* File Upload Area */}
                            <div
                                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${dragActive
                                        ? "border-blue-400 bg-blue-50"
                                        : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
                                    }`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                            >
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileInput}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />

                                <div className="space-y-4">
                                    <div className="p-4 bg-blue-100 rounded-full w-16 h-16 mx-auto flex items-center justify-center">
                                        <Upload className="h-8 w-8 text-blue-600" />
                                    </div>
                                    <div>
                                        <p className="text-lg font-medium text-gray-700">
                                            Drop your image here, or click to
                                            browse
                                        </p>
                                        <p className="text-sm text-gray-500 mt-2">
                                            Supports JPG, PNG, WebP up to 10MB
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Preview */}
                            {previewUrl && (
                                <div className="mt-6">
                                    <div
                                        ref={overlayContainerRef}
                                        className="relative w-full h-64"
                                    >
                                        <img
                                            ref={previewImgRef}
                                            src={previewUrl}
                                            alt="Preview"
                                            className="absolute inset-0 w-full h-full object-contain rounded-xl border border-gray-200 cursor-pointer hover:opacity-90 transition-opacity"
                                            onClick={() => setZoomedImage(previewUrl)}
                                        />
                                        {/* overlay canvas removed; backend returns images */}
                                        <button
                                            onClick={resetUpload}
                                            className="absolute top-3 right-3 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                                        >
                                            <X className="h-4 w-4" />
                                        </button>
                                    </div>

                                    <div className="flex space-x-3 mt-4">
                                        <button
                                            onClick={processImage}
                                            disabled={isProcessing}
                                            className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center"
                                        >
                                            {isProcessing ? (
                                                <>
                                                    <Loader className="animate-spin h-5 w-5 mr-2" />
                                                    Processing...
                                                </>
                                            ) : (
                                                <>
                                                    <FileImage className="h-5 w-5 mr-2" />
                                                    Detect Oil Spills
                                                </>
                                            )}
                                        </button>
                                        <button
                                            onClick={resetUpload}
                                            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors"
                                        >
                                            Reset
                                        </button>
                                    </div>

                                    {/* Auto-detection info */}
                                    <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <p className="text-sm text-gray-600">
                                                    Auto-detected source:
                                                </p>
                                                <p className="text-md font-medium text-gray-900">
                                                    {detectedSource === "sar"
                                                        ? "SAR"
                                                        : detectedSource === "aerial"
                                                            ? "AERIAL (not supported)"
                                                            : "Not detected"}
                                                    {detectionConfidence !==
                                                        null && (
                                                            <span className="text-sm text-gray-500 ml-2">
                                                                (
                                                                {Math.round(
                                                                    detectionConfidence *
                                                                    100,
                                                                )}
                                                                %)
                                                            </span>
                                                        )}
                                                </p>
                                                {detectedSource === "aerial" && (
                                                    <p className="text-xs text-red-600 mt-1">
                                                        Aerial images are not supported. Please upload a SAR image.
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Error Message */}
                            {error && (
                                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start">
                                    <AlertCircle className="h-5 w-5 text-red-500 mr-3 mt-0.5 flex-shrink-0" />
                                    <p className="text-red-700 text-sm">
                                        {error}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Info Panel */}
                        <div className="bg-gradient-to-r from-teal-50 to-blue-50 rounded-2xl p-6 border border-teal-100">
                            <h3 className="font-semibold text-gray-900 mb-3">
                                How it works
                            </h3>
                            <ul className="space-y-2 text-sm text-gray-600">
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5 flex-shrink-0" />
                                    Upload SAR (Synthetic Aperture Radar) images of ocean
                                    areas
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5 flex-shrink-0" />
                                    Both UNet and DeepLabV3+ models are used for
                                    comprehensive analysis
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5 flex-shrink-0" />
                                    AI analyzes the image using advanced
                                    segmentation
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5 flex-shrink-0" />
                                    Receive detailed visualization of potential
                                    oil spills
                                </li>
                            </ul>
                        </div>
                    </div>

                    {/* Results Section */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8">
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-2xl font-semibold text-gray-900">
                                    Detection Results
                                </h2>
                                {resultImageUrl && (
                                    <button
                                        onClick={downloadResult}
                                        className="flex items-center px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
                                    >
                                        <Download className="h-4 w-4 mr-2" />
                                        Download
                                    </button>
                                )}
                            </div>

                            {!resultImageUrl && !isProcessing && (
                                <div className="text-center py-12 text-gray-500">
                                    <FileImage className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                                    <p className="text-lg">
                                        Upload and process an image to see
                                        results
                                    </p>
                                    <p className="text-sm text-gray-400 mt-2">
                                        Upload a SAR image above to begin detection.
                                    </p>
                                </div>
                            )}

                            {isProcessing && (
                                <div className="text-center py-12">
                                    <div className="inline-flex items-center justify-center p-4 bg-blue-100 rounded-full mb-4">
                                        <Loader className="h-8 w-8 text-blue-600 animate-spin" />
                                    </div>
                                    <p className="text-lg text-gray-700">
                                        Processing your image with {getResultModelLabel()}...
                                    </p>
                                    <p className="text-sm text-gray-500 mt-2">
                                        This may take a few seconds
                                    </p>
                                </div>
                            )}

                            {resultImageUrl && (
                                <div className="space-y-6">
                                    <div className="space-y-3">
                                        <h3 className="font-semibold text-gray-900 flex items-center">
                                            <FileImage className="h-5 w-5 mr-2 text-blue-600" />
                                            Analysis Results -{" "}
                                            {getResultModelLabel()}
                                        </h3>
                                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                                            <img
                                                src={resultImageUrl}
                                                alt="Detection Results"
                                                className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                                                onClick={() => setZoomedImage(resultImageUrl)}
                                            />
                                        </div>
                                    </div>

                                    {/* Color Legend */}
                                    <div className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm">
                                        <h4 className="font-semibold text-gray-900 mb-4">
                                            Color Legend
                                        </h4>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                            {COLOR_MAP.map((color, index) => {
                                                const rgbColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                                                return (
                                                    <div
                                                        key={index}
                                                        className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 transition-colors"
                                                    >
                                                        <div
                                                            className="w-8 h-8 rounded-md border border-gray-300 shadow-sm flex-shrink-0"
                                                            style={{
                                                                backgroundColor: rgbColor,
                                                            }}
                                                        />
                                                        <div className="flex-1">
                                                            <p className="text-sm font-medium text-gray-900">
                                                                {CLASS_LABELS[index]}
                                                            </p>
                                                            <p className="text-xs text-gray-500">
                                                                Class {index}
                                                            </p>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    {/* Model Info */}
                                    <div className="bg-gradient-to-r from-blue-50 to-teal-50 rounded-xl p-4 border border-blue-200">
                                        <h4 className="font-semibold text-gray-900 mb-2">
                                            Model Information
                                        </h4>
                                        <p className="text-sm text-gray-600">
                                            Comparison view showing results from both UNet and DeepLabV3+ models side by side. UNet provides excellent precise boundary detection with efficient U-shaped architecture, while DeepLabV3+ offers advanced contextual understanding with dilated convolutions.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
            
            {/* Image Zoom Modal */}
            {zoomedImage && (
                <ImageZoom
                    src={zoomedImage}
                    alt="Zoomed view"
                    onClose={() => setZoomedImage(null)}
                />
            )}
        </div>
    );
}

export default App;
