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

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [resultImageUrl, setResultImageUrl] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [selectedModel, setSelectedModel] = useState("unet");
    const [aerialJson, setAerialJson] = useState(null);

    // Refs for aerial overlay rendering (kept in case we want client-side later)
    const overlayContainerRef = useRef(null);
    const previewImgRef = useRef(null);
    const overlayCanvasRef = useRef(null);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("image/")) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResultImageUrl(null);
            setAerialJson(null);
            setError(null);
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

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const endpoint =
                selectedModel === "both"
                    ? "both"
                    : selectedModel === "deeplab"
                      ? "deeplab"
                      : selectedModel === "aerial"
                        ? "aerial"
                        : "unet";

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
            setResultImageUrl(imageUrl);
            setAerialJson(null);
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
            link.download = `${selectedModel}_prediction.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const resetUpload = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResultImageUrl(null);
        setAerialJson(null);
        setError(null);
        // Clean up object URLs to prevent memory leaks
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        if (resultImageUrl) URL.revokeObjectURL(resultImageUrl);
    };

    // Retain no-op overlay effects (in case we switch back to JSON rendering)
    const drawAerialOverlay = useCallback(() => {}, []);
    useEffect(() => {
        drawAerialOverlay();
    }, [drawAerialOverlay, previewUrl]);
    useEffect(() => {
        const imgEl = previewImgRef.current;
        if (!imgEl) return;
        const handleLoad = () => drawAerialOverlay();
        imgEl.addEventListener("load", handleLoad);
        const handleResize = () => drawAerialOverlay();
        window.addEventListener("resize", handleResize);
        return () => {
            imgEl.removeEventListener("load", handleLoad);
            window.removeEventListener("resize", handleResize);
        };
    }, [drawAerialOverlay]);

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
                                Upload Satellite Image
                            </h2>

                            {/* Model Selection */}
                            <div className="mb-6">
                                <label className="block text-sm font-medium text-gray-700 mb-3">
                                    Select AI Model
                                </label>
                                <div className="grid grid-cols-4 gap-3">
                                    <button
                                        onClick={() => setSelectedModel("unet")}
                                        className={`p-3 rounded-lg border text-sm font-medium transition-all ${
                                            selectedModel === "unet"
                                                ? "bg-blue-600 text-white border-blue-600"
                                                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
                                        }`}
                                    >
                                        UNet
                                    </button>
                                    <button
                                        onClick={() =>
                                            setSelectedModel("deeplab")
                                        }
                                        className={`p-3 rounded-lg border text-sm font-medium transition-all ${
                                            selectedModel === "deeplab"
                                                ? "bg-blue-600 text-white border-blue-600"
                                                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
                                        }`}
                                    >
                                        DeepLabV3+
                                    </button>
                                    <button
                                        onClick={() => setSelectedModel("both")}
                                        className={`p-3 rounded-lg border text-sm font-medium transition-all ${
                                            selectedModel === "both"
                                                ? "bg-blue-600 text-white border-blue-600"
                                                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
                                        }`}
                                    >
                                        Both Models
                                    </button>
                                    <button
                                        onClick={() =>
                                            setSelectedModel("aerial")
                                        }
                                        className={`p-3 rounded-lg border text-sm font-medium transition-all ${
                                            selectedModel === "aerial"
                                                ? "bg-blue-600 text-white border-blue-600"
                                                : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
                                        }`}
                                    >
                                        Aerial (Roboflow)
                                    </button>
                                </div>
                            </div>

                            {/* File Upload Area */}
                            <div
                                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                                    dragActive
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
                                            className="absolute inset-0 w-full h-full object-contain rounded-xl border border-gray-200"
                                        />
                                        {/* Keep overlay canvas hidden for now since backend returns images */}
                                        {false && (
                                            <canvas
                                                ref={overlayCanvasRef}
                                                className="absolute inset-0 w-full h-full pointer-events-none rounded-xl"
                                            />
                                        )}
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
                                    Upload satellite or aerial images of ocean
                                    areas
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5 flex-shrink-0" />
                                    Choose between UNet, DeepLabV3+, or both
                                    models
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
                                        Select a model and upload your satellite
                                        image above
                                    </p>
                                </div>
                            )}

                            {isProcessing && (
                                <div className="text-center py-12">
                                    <div className="inline-flex items-center justify-center p-4 bg-blue-100 rounded-full mb-4">
                                        <Loader className="h-8 w-8 text-blue-600 animate-spin" />
                                    </div>
                                    <p className="text-lg text-gray-700">
                                        Processing your image with{" "}
                                        {selectedModel === "both"
                                            ? "both models"
                                            : selectedModel}
                                        ...
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
                                            {selectedModel === "both"
                                                ? "Both Models"
                                                : selectedModel.toUpperCase()}
                                        </h3>
                                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                                            <img
                                                src={resultImageUrl}
                                                alt="Detection Results"
                                                className="w-full h-auto"
                                            />
                                        </div>
                                    </div>

                                    {/* Color Legend */}
                                    {/* <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl p-4 border border-gray-200"> */}
                                    {/*     <h4 className="font-semibold text-gray-900 mb-3"> */}
                                    {/*         Color Legend */}
                                    {/*     </h4> */}
                                    {/*     <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs"> */}
                                    {/*         <div className="flex items-center"> */}
                                    {/*             <div className="w-4 h-4 bg-black border border-gray-300 rounded mr-2"></div> */}
                                    {/*             <span>Background</span> */}
                                    {/*         </div> */}
                                    {/*         <div className="flex items-center"> */}
                                    {/*             <div className="w-4 h-4 bg-cyan-400 border border-gray-300 rounded mr-2"></div> */}
                                    {/*             <span>Water</span> */}
                                    {/*         </div> */}
                                    {/*         <div className="flex items-center"> */}
                                    {/*             <div className="w-4 h-4 bg-red-500 border border-gray-300 rounded mr-2"></div> */}
                                    {/*             <span>Oil Spill</span> */}
                                    {/*         </div> */}
                                    {/*         <div className="flex items-center"> */}
                                    {/*             <div */}
                                    {/*                 className="w-4 h-4" */}
                                    {/*                 style={{ */}
                                    {/*                     backgroundColor: */}
                                    {/*                         "rgb(153, 76, 0)", */}
                                    {/*                 }} */}
                                    {/*             ></div> */}
                                    {/*             <span className="ml-2"> */}
                                    {/*                 Land/Shore */}
                                    {/*             </span> */}
                                    {/*         </div> */}
                                    {/*         <div className="flex items-center"> */}
                                    {/*             <div */}
                                    {/*                 className="w-4 h-4" */}
                                    {/*                 style={{ */}
                                    {/*                     backgroundColor: */}
                                    {/*                         "rgb(0, 153, 0)", */}
                                    {/*                 }} */}
                                    {/*             ></div> */}
                                    {/*             <span className="ml-2"> */}
                                    {/*                 Vegetation */}
                                    {/*             </span> */}
                                    {/*         </div> */}
                                    {/*     </div> */}
                                    {/* </div> */}
                                    {/**/}
                                    {/* Model Info */}
                                    <div className="bg-gradient-to-r from-blue-50 to-teal-50 rounded-xl p-4 border border-blue-200">
                                        <h4 className="font-semibold text-gray-900 mb-2">
                                            Model Information
                                        </h4>
                                        <p className="text-sm text-gray-600">
                                            {selectedModel === "unet" &&
                                                "UNet: Excellent for precise boundary detection with efficient U-shaped architecture."}
                                            {selectedModel === "deeplab" &&
                                                "DeepLabV3+: Advanced model with dilated convolutions for improved contextual understanding."}
                                            {selectedModel === "both" &&
                                                "Comparison view showing results from both UNet and DeepLabV3+ models side by side."}
                                            {selectedModel === "aerial" &&
                                                "Aerial workflow visualization composed server-side from Roboflow polygons."}
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
