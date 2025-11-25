import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image  
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Image = Any  

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    enabled: bool = True
    language: str = "eng"
    dpi: int = 300
    enhance_contrast: bool = True
    tesseract_path: Optional[str] = None
    min_confidence: float = 60.0
    preprocess: bool = True


class OCRError(Exception):
    """Exception raised for OCR errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class OCRProcessor:
    """
    OCR processor for extracting text from images and scanned documents.
    
    Uses Tesseract OCR (via pytesseract) for text extraction.
    Supports image preprocessing for better accuracy.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize the OCR processor.
        
        Args:
            config: OCR configuration
        """
        self.config = config or OCRConfig()
        self._tesseract_available = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize OCR engine."""
        if not self.config.enabled:
            logger.info("OCR is disabled")
            return
        
        try:
            import pytesseract
            
            # Set Tesseract path if provided
            if self.config.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
            
            # Verify Tesseract is available
            version = pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info(f"Tesseract OCR initialized (version: {version})")
            
        except ImportError:
            logger.warning(
                "pytesseract not available. Install with: pip install pytesseract"
            )
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.config.enabled and self._tesseract_available
    
    def _load_image(self, image_path: str):
        """Load an image file."""
        try:
            from PIL import Image
            return Image.open(image_path)
        except ImportError:
            raise OCRError("Pillow is required. Install with: pip install Pillow")
        except Exception as e:
            raise OCRError(f"Failed to load image: {str(e)}", e)
    
    def _preprocess_image(self, image) -> "Image":
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import numpy as np
        except ImportError:
            return image
        
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")
        
        # Enhance contrast
        if self.config.enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small
        min_dimension = 1000
        width, height = image.size
        if width < min_dimension or height < min_dimension:
            scale = max(min_dimension / width, min_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text_from_image(
        self,
        image_path: str,
        preprocess: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess (uses config if None)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.is_available:
            return {
                "text": "",
                "confidence": 0.0,
                "error": "OCR not available",
                "success": False
            }
        
        try:
            import pytesseract
            from PIL import Image
            
            # Load image
            image = self._load_image(image_path)
            
            # Preprocess if enabled
            should_preprocess = preprocess if preprocess is not None else self.config.preprocess
            if should_preprocess:
                image = self._preprocess_image(image)
            
            # Perform OCR
            custom_config = f"--oem 3 --psm 3 -l {self.config.language}"
            
            # Get text with confidence data
            data = pytesseract.image_to_data(
                image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter by confidence and extract text
            texts = []
            confidences = []
            
            for i, text in enumerate(data["text"]):
                conf = float(data["conf"][i])
                if text.strip() and conf >= self.config.min_confidence:
                    texts.append(text)
                    confidences.append(conf)
            
            extracted_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(
                f"OCR extracted {len(extracted_text)} chars from {image_path} "
                f"(confidence: {avg_confidence:.1f}%)"
            )
            
            return {
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": len(texts),
                "source": image_path,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "source": image_path,
                "success": False
            }
    
    def extract_text_from_pdf_images(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from images embedded in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            pages: Specific pages to process (all if None)
            
        Returns:
            List of extraction results per page
        """
        if not self.is_available:
            return [{
                "page": 0,
                "text": "",
                "error": "OCR not available",
                "success": False
            }]
        
        results = []
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            pages_to_process = pages if pages else list(range(total_pages))
            
            for page_num in pages_to_process:
                if page_num >= total_pages:
                    continue
                
                page = doc[page_num]
                
                # Render page as image
                pix = page.get_pixmap(dpi=self.config.dpi)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pix.save(tmp.name)
                    tmp_path = tmp.name
                
                try:
                    # Extract text
                    result = self.extract_text_from_image(tmp_path)
                    result["page"] = page_num + 1
                    results.append(result)
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            doc.close()
            
            logger.info(f"OCR processed {len(results)} pages from {pdf_path}")
            return results
            
        except ImportError:
            logger.error("PyMuPDF is required for PDF OCR. Install with: pip install PyMuPDF")
            return [{
                "page": 0,
                "text": "",
                "error": "PyMuPDF not available",
                "success": False
            }]
        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            return [{
                "page": 0,
                "text": "",
                "error": str(e),
                "success": False
            }]
    
    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Detect if a PDF is scanned (image-based) vs. text-based.
        
        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample
            
        Returns:
            True if PDF appears to be scanned/image-based
        """
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            pages_to_check = min(sample_pages, len(doc))
            
            text_char_counts = []
            image_counts = []
            
            for i in range(pages_to_check):
                page = doc[i]
                
                # Count text characters
                text = page.get_text()
                text_char_counts.append(len(text.strip()))
                
                # Count images
                images = page.get_images()
                image_counts.append(len(images))
            
            doc.close()
            
            # If average text is very low and images exist, likely scanned
            avg_text = sum(text_char_counts) / len(text_char_counts) if text_char_counts else 0
            has_images = sum(image_counts) > 0
            
            # Threshold: less than 100 chars per page average suggests scanned
            is_scanned = avg_text < 100 and has_images
            
            logger.debug(
                f"PDF scan detection: avg_text={avg_text:.0f}, "
                f"has_images={has_images}, is_scanned={is_scanned}"
            )
            
            return is_scanned
            
        except Exception as e:
            logger.warning(f"Could not detect if PDF is scanned: {str(e)}")
            return False
    
    def process_document(
        self,
        file_path: str,
        existing_content: str = ""
    ) -> Dict[str, Any]:
        """
        Process a document with OCR if needed.
        
        Args:
            file_path: Path to the document
            existing_content: Already extracted text content
            
        Returns:
            Dictionary with processed content
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        result = {
            "original_content": existing_content,
            "ocr_content": "",
            "combined_content": existing_content,
            "ocr_applied": False,
            "ocr_pages": [],
            "success": True
        }
        
        if not self.is_available:
            return result
        
        # Handle image files
        if extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
            ocr_result = self.extract_text_from_image(file_path)
            result["ocr_content"] = ocr_result.get("text", "")
            result["combined_content"] = ocr_result.get("text", "")
            result["ocr_applied"] = ocr_result.get("success", False)
            result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            return result
        
        # Handle PDFs
        if extension == ".pdf":
            # Check if it's a scanned PDF
            if self.is_scanned_pdf(file_path) or len(existing_content.strip()) < 100:
                ocr_results = self.extract_text_from_pdf_images(file_path)
                
                ocr_texts = []
                for page_result in ocr_results:
                    if page_result.get("success") and page_result.get("text"):
                        ocr_texts.append(page_result["text"])
                        result["ocr_pages"].append(page_result.get("page", 0))
                
                result["ocr_content"] = "\n\n".join(ocr_texts)
                result["ocr_applied"] = len(ocr_texts) > 0
                
                # Combine or replace
                if result["ocr_content"] and len(result["ocr_content"]) > len(existing_content):
                    result["combined_content"] = result["ocr_content"]
                else:
                    result["combined_content"] = existing_content or result["ocr_content"]
        
        return result


def create_ocr_processor(
    language: str = "eng",
    **kwargs
) -> OCRProcessor:
    """
    Factory function to create an OCR processor.
    
    Args:
        language: OCR language code
        **kwargs: Additional configuration options
        
    Returns:
        Configured OCRProcessor instance
    """
    config = OCRConfig(language=language, **kwargs)
    return OCRProcessor(config)

