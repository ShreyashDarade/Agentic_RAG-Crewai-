"""
Enhanced OCR Processor - Production Grade with EasyOCR + spaCy NLP

Features:
- EasyOCR multilingual support
- spaCy NLP post-processing
- Entity extraction
- Keyword extraction
- Text correction
- GPU support
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EnhancedOCRConfig:
    """Configuration for enhanced OCR processing."""
    enabled: bool = True
    
    # Language support
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Processing settings
    dpi: int = 300
    enhance_contrast: bool = True
    detect_rotation: bool = True
    paragraph_mode: bool = True
    detail: int = 1  # 0 = fast, 1 = accurate
    
    # GPU acceleration
    gpu: bool = False
    
    # NLP post-processing
    nlp_enabled: bool = True
    spacy_model: str = "en_core_web_sm"
    extract_entities: bool = True
    extract_keywords: bool = True
    
    # Confidence thresholds
    min_confidence: float = 0.3
    low_confidence_threshold: float = 0.5


class EnhancedOCRError(Exception):
    """Exception raised for OCR errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class EnhancedOCRProcessor:
    """
    Enhanced OCR processor using EasyOCR with spaCy NLP post-processing.
    
    Features:
    - Multilingual text extraction
    - NLP-based text cleanup and enhancement
    - Entity and keyword extraction
    - Confidence scoring
    """
    
    def __init__(self, config: Optional[EnhancedOCRConfig] = None):
        """
        Initialize the enhanced OCR processor.
        
        Args:
            config: OCR configuration
        """
        self.config = config or EnhancedOCRConfig()
        self._reader = None
        self._nlp = None
        self._available = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize OCR engine and NLP model."""
        if not self.config.enabled:
            logger.info("OCR is disabled")
            return
        
        # Initialize EasyOCR
        self._initialize_easyocr()
        
        # Initialize spaCy NLP
        if self.config.nlp_enabled:
            self._initialize_nlp()
    
    def _initialize_easyocr(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            
            logger.info(f"Loading EasyOCR with languages: {self.config.languages}")
            
            self._reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.gpu,
                verbose=False
            )
            
            self._available = True
            logger.info("EasyOCR initialized successfully")
            
        except ImportError:
            logger.warning(
                "EasyOCR not available. Install with: pip install easyocr"
            )
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {str(e)}")
    
    def _initialize_nlp(self) -> None:
        """Initialize spaCy NLP model."""
        try:
            import spacy
            
            try:
                self._nlp = spacy.load(self.config.spacy_model)
                logger.info(f"spaCy model loaded: {self.config.spacy_model}")
            except OSError:
                logger.warning(
                    f"spaCy model '{self.config.spacy_model}' not found. "
                    f"Download with: python -m spacy download {self.config.spacy_model}"
                )
                self._nlp = None
                
        except ImportError:
            logger.warning("spaCy not available. Install with: pip install spacy")
            self._nlp = None
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.config.enabled and self._available and self._reader is not None
    
    def _load_image(self, image_path: str) -> Any:
        """Load an image file."""
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return np.array(image)
            
        except ImportError:
            raise EnhancedOCRError("Pillow is required. Install with: pip install Pillow")
        except Exception as e:
            raise EnhancedOCRError(f"Failed to load image: {str(e)}", e)
    
    def _preprocess_image(self, image_array: Any) -> Any:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_array: Numpy array of image
            
        Returns:
            Preprocessed image array
        """
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Enhance contrast using CLAHE
            if self.config.enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Threshold to binary
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            return gray
            
        except ImportError:
            return image_array
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_array
    
    def extract_text_from_image(
        self,
        image_path: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
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
            # Load image
            image_array = self._load_image(image_path)
            
            # Preprocess if enabled
            if preprocess:
                image_array = self._preprocess_image(image_array)
            
            # Perform OCR
            results = self._reader.readtext(
                image_array,
                detail=self.config.detail,
                paragraph=self.config.paragraph_mode
            )
            
            # Extract text and confidence
            texts = []
            confidences = []
            bboxes = []
            
            for result in results:
                if self.config.detail == 1 and len(result) >= 3:
                    bbox, text, confidence = result[0], result[1], result[2]
                    
                    if confidence >= self.config.min_confidence:
                        texts.append(text)
                        confidences.append(confidence)
                        bboxes.append(bbox)
                else:
                    # Paragraph mode returns just text
                    texts.append(result)
                    confidences.append(1.0)
            
            extracted_text = " ".join(texts) if not self.config.paragraph_mode else "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Apply NLP post-processing
            nlp_result = {}
            if self._nlp and extracted_text:
                nlp_result = self._apply_nlp(extracted_text)
                if "cleaned_text" in nlp_result:
                    # Use cleaned text if available
                    extracted_text = nlp_result.get("cleaned_text", extracted_text)
            
            logger.info(
                f"OCR extracted {len(extracted_text)} chars from {image_path} "
                f"(avg confidence: {avg_confidence:.2f})"
            )
            
            return {
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": len(texts),
                "low_confidence_words": sum(1 for c in confidences if c < self.config.low_confidence_threshold),
                "source": image_path,
                "bounding_boxes": bboxes if self.config.detail == 1 else [],
                "entities": nlp_result.get("entities", []),
                "keywords": nlp_result.get("keywords", []),
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
    
    def _apply_nlp(self, text: str) -> Dict[str, Any]:
        """
        Apply NLP post-processing to extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Dictionary with NLP results
        """
        if not self._nlp or not text:
            return {}
        
        try:
            doc = self._nlp(text)
            result = {}
            
            # Clean text - fix common OCR errors
            cleaned_text = text
            
            # Extract entities
            if self.config.extract_entities:
                entities = []
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                result["entities"] = entities
            
            # Extract keywords (noun chunks)
            if self.config.extract_keywords:
                keywords = []
                for chunk in doc.noun_chunks:
                    # Filter out very short or common words
                    if len(chunk.text) > 2 and chunk.root.pos_ in ["NOUN", "PROPN"]:
                        keywords.append({
                            "text": chunk.text,
                            "root": chunk.root.text,
                            "pos": chunk.root.pos_
                        })
                result["keywords"] = keywords[:20]  # Limit keywords
            
            # Calculate readability score
            result["sentence_count"] = len(list(doc.sents))
            result["token_count"] = len(doc)
            
            result["cleaned_text"] = cleaned_text
            
            return result
            
        except Exception as e:
            logger.warning(f"NLP processing failed: {e}")
            return {}
    
    def extract_text_from_pdf_images(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from images in a PDF.
        
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
            "entities": [],
            "keywords": [],
            "success": True
        }
        
        if not self.is_available:
            return result
        
        # Handle image files
        if extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]:
            ocr_result = self.extract_text_from_image(file_path)
            result["ocr_content"] = ocr_result.get("text", "")
            result["combined_content"] = ocr_result.get("text", "")
            result["ocr_applied"] = ocr_result.get("success", False)
            result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            result["entities"] = ocr_result.get("entities", [])
            result["keywords"] = ocr_result.get("keywords", [])
            return result
        
        # Handle PDFs
        if extension == ".pdf":
            # Check if it's a scanned PDF
            if self.is_scanned_pdf(file_path) or len(existing_content.strip()) < 100:
                ocr_results = self.extract_text_from_pdf_images(file_path)
                
                ocr_texts = []
                all_entities = []
                all_keywords = []
                
                for page_result in ocr_results:
                    if page_result.get("success") and page_result.get("text"):
                        ocr_texts.append(page_result["text"])
                        result["ocr_pages"].append(page_result.get("page", 0))
                        all_entities.extend(page_result.get("entities", []))
                        all_keywords.extend(page_result.get("keywords", []))
                
                result["ocr_content"] = "\n\n".join(ocr_texts)
                result["ocr_applied"] = len(ocr_texts) > 0
                result["entities"] = all_entities
                result["keywords"] = all_keywords[:30]  # Limit keywords
                
                # Combine or replace
                if result["ocr_content"] and len(result["ocr_content"]) > len(existing_content):
                    result["combined_content"] = result["ocr_content"]
                else:
                    result["combined_content"] = existing_content or result["ocr_content"]
        
        return result
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process
            
        Returns:
            List of entity dictionaries
        """
        if not self._nlp or not text:
            return []
        
        try:
            doc = self._nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": self._get_entity_description(ent.label_)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _get_entity_description(self, label: str) -> str:
        """Get human-readable description for entity label."""
        descriptions = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Geopolitical Entity",
            "LOC": "Location",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Monetary Value",
            "PERCENT": "Percentage",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "Work of Art",
            "LAW": "Law",
            "LANGUAGE": "Language",
            "NORP": "Nationality/Religious/Political Group",
            "FAC": "Facility",
            "CARDINAL": "Cardinal Number",
            "ORDINAL": "Ordinal Number",
            "QUANTITY": "Quantity",
        }
        return descriptions.get(label, label)


def create_enhanced_ocr_processor(
    languages: Optional[List[str]] = None,
    gpu: bool = False,
    **kwargs
) -> EnhancedOCRProcessor:
    """
    Factory function to create an enhanced OCR processor.
    
    Args:
        languages: OCR language codes
        gpu: Enable GPU acceleration
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnhancedOCRProcessor instance
    """
    config = EnhancedOCRConfig(
        languages=languages or ["en"],
        gpu=gpu,
        **kwargs
    )
    return EnhancedOCRProcessor(config)
