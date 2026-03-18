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
    const DEFAULT_MODEL = "deeplab"; // 'deeplab' | 'unet' | 'both'
    const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
    const [zoomedImage, setZoomedImage] = useState(null); // URL of image to zoom
    const [detectionInfo, setDetectionInfo] = useState(null); // oil spill detection summary

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
    const modelSettingsRef = useRef(null);

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
            setDetectionInfo(null);
            setError(null);
            // Reset model selection back to default when a new file is chosen
            setSelectedModel(DEFAULT_MODEL);
        } else {
            setError("Please select a valid image file");
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

        setIsProcessing(true);
        setError(null);
        setResultImageUrl(null);
        setDetectionInfo(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            // Choose backend endpoint based on selected model
            let endpoint = "deeplab";
            if (selectedModel === "unet") {
                endpoint = "unet";
            } else if (selectedModel === "both") {
                endpoint = "both";
            }

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

            // Parse oil-spill detection headers from the backend
            const parseBool = (value) =>
                typeof value === "string" &&
                value.toLowerCase() === "true";

            let detectionSummary = null;

            if (selectedModel === "both") {
                const unetPresent = parseBool(
                    response.headers.get("x-oil-spill-present-unet"),
                );
                const unetFraction = parseFloat(
                    response.headers.get("x-oil-spill-fraction-unet") || "0",
                );
                const deeplabPresent = parseBool(
                    response.headers.get("x-oil-spill-present-deeplab"),
                );
                const deeplabFraction = parseFloat(
                    response.headers.get("x-oil-spill-fraction-deeplab") || "0",
                );
                const unetAreaKm2 = parseFloat(
                    response.headers.get("x-oil-spill-area-km2-unet") || "0",
                );
                const deeplabAreaKm2 = parseFloat(
                    response.headers.get("x-oil-spill-area-km2-deeplab") || "0",
                );

                if (
                    !Number.isNaN(unetFraction) ||
                    !Number.isNaN(deeplabFraction)
                ) {
                    const models = [
                        {
                            name: "UNet",
                            hasSpill: !!unetPresent,
                            fraction: Number.isNaN(unetFraction)
                                ? 0
                                : unetFraction,
                            areaKm2: Number.isNaN(unetAreaKm2)
                                ? 0
                                : unetAreaKm2,
                        },
                        {
                            name: "DeepLabV3+",
                            hasSpill: !!deeplabPresent,
                            fraction: Number.isNaN(deeplabFraction)
                                ? 0
                                : deeplabFraction,
                            areaKm2: Number.isNaN(deeplabAreaKm2)
                                ? 0
                                : deeplabAreaKm2,
                        },
                    ];

                    const overallHasSpill = models.some(
                        (m) => m.hasSpill,
                    );

                    detectionSummary = {
                        overallHasSpill,
                        models,
                    };
                }
            } else {
                const present = parseBool(
                    response.headers.get("x-oil-spill-present"),
                );
                const fraction = parseFloat(
                    response.headers.get("x-oil-spill-fraction") || "0",
                );
                const areaKm2 = parseFloat(
                    response.headers.get("x-oil-spill-area-km2") || "0",
                );

                if (!Number.isNaN(fraction)) {
                    detectionSummary = {
                        overallHasSpill: !!present,
                        models: [
                            {
                                name: getResultModelLabel(),
                                hasSpill: !!present,
                                fraction: Number.isNaN(fraction)
                                    ? 0
                                    : fraction,
                                areaKm2: Number.isNaN(areaKm2)
                                    ? 0
                                    : areaKm2,
                            },
                        ],
                    };
                }
            }

            if (detectionSummary) {
                setDetectionInfo(detectionSummary);
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
        if (selectedModel === "unet") return "UNet";
        if (selectedModel === "both") return "UNet + DeepLabV3+";
        return "DeepLabV3+";
    };

    const resetUpload = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResultImageUrl(null);
        setError(null);
        setSelectedModel(DEFAULT_MODEL);
        setDetectionInfo(null);
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
                        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8 relative">
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-2xl font-semibold text-gray-900 flex items-center">
                                    <Upload className="h-6 w-6 mr-3 text-blue-600" />
                                    Upload SAR Image
                                </h2>
                                {/* Small, subtle settings trigger in the corner */}
                                <details className="relative" ref={modelSettingsRef}>
                                    <summary className="list-none text-xs text-gray-500 hover:text-gray-700 cursor-pointer px-2 py-1 rounded-md hover:bg-gray-100 transition-colors">
                                        Model settings
                                    </summary>
                                    <div className="absolute right-0 mt-2 w-72 bg-white border border-gray-200 rounded-xl shadow-lg p-4 z-20">
                                        <div className="space-y-2 text-sm text-gray-800">
                                            <label className="flex items-center space-x-2 cursor-pointer">
                                                <input
                                                    type="radio"
                                                    name="model"
                                                    value="deeplab"
                                                    checked={selectedModel === "deeplab"}
                                                    onChange={() => {
                                                        setSelectedModel("deeplab");
                                                        if (modelSettingsRef.current) {
                                                            modelSettingsRef.current.open = false;
                                                        }
                                                    }}
                                                    className="text-blue-600 focus:ring-blue-500"
                                                />
                                                <span>DeepLabV3+ (default)</span>
                                            </label>
                                            <label className="flex items-center space-x-2 cursor-pointer">
                                                <input
                                                    type="radio"
                                                    name="model"
                                                    value="unet"
                                                    checked={selectedModel === "unet"}
                                                    onChange={() => {
                                                        setSelectedModel("unet");
                                                        if (modelSettingsRef.current) {
                                                            modelSettingsRef.current.open = false;
                                                        }
                                                    }}
                                                    className="text-blue-600 focus:ring-blue-500"
                                                />
                                                <span>UNet</span>
                                            </label>
                                            <label className="flex items-center space-x-2 cursor-pointer">
                                                <input
                                                    type="radio"
                                                    name="model"
                                                    value="both"
                                                    checked={selectedModel === "both"}
                                                    onChange={() => {
                                                        setSelectedModel("both");
                                                        if (modelSettingsRef.current) {
                                                            modelSettingsRef.current.open = false;
                                                        }
                                                    }}
                                                    className="text-blue-600 focus:ring-blue-500"
                                                />
                                                <span>UNet and DeepLabV3+</span>
                                            </label>
                                        </div>
                                    </div>
                                </details>
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

                            {/* Oil Spill Detection Summary */}
                            {detectionInfo && !isProcessing && resultImageUrl && (
                                <div
                                    className={`mb-6 p-4 rounded-xl border flex items-start space-x-3 ${
                                        detectionInfo.overallHasSpill
                                            ? "bg-red-50 border-red-200"
                                            : "bg-green-50 border-green-200"
                                    }`}
                                >
                                    <div className="mt-0.5">
                                        {detectionInfo.overallHasSpill ? (
                                            <AlertCircle className="h-5 w-5 text-red-600" />
                                        ) : (
                                            <CheckCircle className="h-5 w-5 text-green-600" />
                                        )}
                                    </div>
                                    <div>
                                        <p
                                            className={`font-semibold text-sm ${
                                                detectionInfo.overallHasSpill
                                                    ? "text-red-800"
                                                    : "text-green-800"
                                            }`}
                                        >
                                            {detectionInfo.overallHasSpill
                                                ? "Oil contamination detected"
                                                : "No oil contamination detected"}
                                        </p>
                                        {detectionInfo.models &&
                                            detectionInfo.models.length > 0 && (
                                                <ul className="mt-2 space-y-1 text-xs text-gray-700">
                                                    {detectionInfo.models.map(
                                                        (m, idx) => (
                                                            <li key={idx}>
                                                                <span className="font-medium">
                                                                    {m.name}
                                                                    {": "}
                                                                </span>
                                                                {m.hasSpill
                                                                    ? m.areaKm2 && m.areaKm2 > 0
                                                                        ? `spill detected (~${m.areaKm2.toFixed(
                                                                              3,
                                                                          )} km²)`
                                                                        : "spill detected"
                                                                    : "no spill region detected"}
                                                            </li>
                                                        ),
                                                    )}
                                                </ul>
                                            )}
                                    </div>
                                </div>
                            )}

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
                                        {selectedModel === "deeplab" && (
                                            <p className="text-sm text-gray-600">
                                                DeepLabV3+ provides strong contextual understanding using atrous (dilated)
                                                convolutions and an encoder-decoder design, making it well-suited for
                                                detailed oil spill segmentation from SAR imagery.
                                            </p>
                                        )}
                                        {selectedModel === "unet" && (
                                            <p className="text-sm text-gray-600">
                                                UNet uses a U-shaped encoder-decoder architecture with skip connections,
                                                giving precise boundary localization and efficient segmentation for oil
                                                spill detection.
                                            </p>
                                        )}
                                        {selectedModel === "both" && (
                                            <p className="text-sm text-gray-600">
                                                Comparison view combining UNet and DeepLabV3+ predictions side by side.
                                                UNet offers sharp boundary detection with its U-shaped architecture, while
                                                DeepLabV3+ adds rich contextual understanding via atrous convolutions.
                                            </p>
                                        )}
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
