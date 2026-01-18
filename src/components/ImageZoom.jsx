import { X, ZoomIn } from "lucide-react";

const ImageZoom = ({ src, alt, onClose }) => {
    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
            onClick={onClose}
        >
            <button
                onClick={onClose}
                className="absolute top-4 right-4 p-2 bg-white/10 hover:bg-white/20 rounded-full text-white transition-colors z-10"
                aria-label="Close"
            >
                <X className="h-6 w-6" />
            </button>
            <div className="relative max-w-[95vw] max-h-[95vh] p-4" onClick={(e) => e.stopPropagation()}>
                <img
                    src={src}
                    alt={alt}
                    className="max-w-full max-h-[95vh] object-contain rounded-lg shadow-2xl"
                />
            </div>
        </div>
    );
};

export default ImageZoom;
