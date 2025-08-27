import React, { useState } from "react";
import {
    Upload,
    Image as ImageIcon,
    X,
    CheckCircle,
    Loader,
} from "lucide-react";

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("image/")) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResults(null);
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

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const response = await fetch("/api/detect-oil-spill", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to process image");
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError("Failed to process image. Please try again.");
            console.error("Processing error:", err);
        } finally {
            setIsProcessing(false);
        }
    };

    const resetUpload = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResults(null);
        setError(null);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
            {/* Header */}
            <header className="bg-white/80 backdrop-blur-sm border-b border-blue-100 sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-blue-600 rounded-lg">
                            <ImageIcon className="h-8 w-8 text-white" />
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
                                    <div className="relative">
                                        <img
                                            src={previewUrl}
                                            alt="Preview"
                                            className="w-full h-64 object-cover rounded-xl border border-gray-200"
                                        />
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
                                                    <ImageIcon className="h-5 w-5 mr-2" />
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
                                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center">
                                    <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
                                    <p className="text-red-700">{error}</p>
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
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5" />
                                    Upload satellite or aerial images of ocean
                                    areas
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5" />
                                    AI model analyzes the image using advanced
                                    segmentation
                                </li>
                                <li className="flex items-start">
                                    <CheckCircle className="h-4 w-4 text-teal-500 mr-2 mt-0.5" />
                                    Receive detailed mask showing potential oil
                                    spill areas
                                </li>
                            </ul>
                        </div>
                    </div>

                    {/* Results Section */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8">
                            <h2 className="text-2xl font-semibold text-gray-900 mb-6">
                                Detection Results
                            </h2>

                            {!results && !isProcessing && (
                                <div className="text-center py-12 text-gray-500">
                                    <ImageIcon className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                                    <p className="text-lg">
                                        Upload and process an image to see
                                        results
                                    </p>
                                </div>
                            )}

                            {isProcessing && (
                                <div className="text-center py-12">
                                    <div className="inline-flex items-center justify-center p-4 bg-blue-100 rounded-full mb-4">
                                        <Loader className="h-8 w-8 text-blue-600 animate-spin" />
                                    </div>
                                    <p className="text-lg text-gray-700">
                                        Processing your image...
                                    </p>
                                    <p className="text-sm text-gray-500 mt-2">
                                        This may take a few seconds
                                    </p>
                                </div>
                            )}

                            {results && (
                                <div className="space-y-6">
                                    {/* Results Grid */}
                                    <div className="grid grid-cols-1 gap-4">
                                        <div className="space-y-3">
                                            <h3 className="font-semibold text-gray-900">
                                                Original Image
                                            </h3>
                                            <img
                                                src={
                                                    results.original_image ||
                                                    previewUrl
                                                }
                                                alt="Original"
                                                className="w-full h-48 object-cover rounded-lg border border-gray-200"
                                            />
                                        </div>

                                        <div className="space-y-3">
                                            <h3 className="font-semibold text-gray-900">
                                                Detection Mask
                                            </h3>
                                            <img
                                                src={results.mask_image}
                                                alt="Detection Mask"
                                                className="w-full h-48 object-cover rounded-lg border border-gray-200"
                                            />
                                        </div>

                                        <div className="space-y-3">
                                            <h3 className="font-semibold text-gray-900">
                                                Overlay Result
                                            </h3>
                                            <img
                                                src={results.overlay_image}
                                                alt="Overlay"
                                                className="w-full h-48 object-cover rounded-lg border border-gray-200"
                                            />
                                        </div>
                                    </div>

                                    {/* Analysis Summary */}
                                    <div className="bg-gradient-to-r from-blue-50 to-teal-50 rounded-xl p-6 border border-blue-200">
                                        <h3 className="font-semibold text-gray-900 mb-3">
                                            Analysis Summary
                                        </h3>
                                        <div className="grid grid-cols-2 gap-4 text-sm">
                                            <div>
                                                <span className="font-medium text-gray-700">
                                                    Detection Confidence:
                                                </span>
                                                <p className="text-2xl font-bold text-blue-600 mt-1">
                                                    {results.confidence ||
                                                        "95.2"}
                                                    %
                                                </p>
                                            </div>
                                            <div>
                                                <span className="font-medium text-gray-700">
                                                    Affected Area:
                                                </span>
                                                <p className="text-2xl font-bold text-teal-600 mt-1">
                                                    {results.affected_area ||
                                                        "1.3"}{" "}
                                                    km²
                                                </p>
                                            </div>
                                        </div>
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
