"""
Donut (Document Understanding Transformer) Engine Implementation
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from .base import BaseOCREngine, OCRResult

logger = get_logger(__name__)


class DonutEngine(BaseOCREngine):
    """Donut-based document understanding engine."""

    name = "donut"
    description = "Donut - Document Understanding Transformer (end-to-end)"

    # Hugging Face model name
    MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize Donut engine.

        Args:
            model_name: Hugging Face model name (default: CORD receipt model)
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__()
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._processor = None

    @classmethod
    def _check_availability(cls) -> bool:
        """
        Check if Donut is available.

        NOTE: Donut requires ~1.5GB model download and GPU for practical use.
        Disabled for local development. Enable for Colab/Cloud testing.
        """
        # Disabled: Heavy model (~1.5GB), requires GPU for practical speed
        # To enable: return True and ensure torch + transformers are installed
        return False

    def _initialize(self) -> None:
        """Initialize Donut model and processor."""
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        logger.info(f"Loading Donut model: {self.model_name}...")

        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = DonutProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        logger.info(f"Donut ready on {self.device}.")

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using Donut model."""
        import torch
        from PIL import Image
        import re
        import json

        # Load and process image
        image = Image.open(image_path).convert("RGB")

        # Prepare input
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self._processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids

        pixel_values = self._processor(image, return_tensors="pt").pixel_values

        # Move to device
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode output
        sequence = self._processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self._processor.tokenizer.eos_token, "").replace(self._processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        # Try to parse as JSON (Donut outputs structured data)
        parsed_data = {}
        try:
            # Donut CORD model outputs token-based format
            parsed_data = self._parse_donut_output(sequence)
        except Exception as e:
            logger.debug(f"Could not parse Donut output: {e}")

        # Extract fields from parsed data
        fields = self._extract_fields_from_parsed(parsed_data, sequence)

        # Calculate confidence (Donut doesn't provide per-token confidence)
        # Use a heuristic based on successful parsing
        confidence = 0.8 if fields.get('total') else 0.5

        return OCRResult(
            raw_text=sequence,
            lines=[{'text': sequence, 'confidence': confidence}],
            confidence=confidence,
            fields=fields,
            metadata={
                'model': self.model_name,
                'device': self.device,
                'parsed_data': parsed_data
            }
        )

    def _parse_donut_output(self, sequence: str) -> Dict[str, Any]:
        """Parse Donut's token-based output format."""
        import re

        result = {}

        # Extract menu items
        menu_pattern = r'<s_menu><s_nm>(.*?)</s_nm>.*?<s_price>(.*?)</s_price>'
        menu_matches = re.findall(menu_pattern, sequence)
        if menu_matches:
            result['items'] = [{'name': m[0], 'price': m[1]} for m in menu_matches]

        # Extract total
        total_pattern = r'<s_total>.*?<s_total_price>(.*?)</s_total_price>'
        total_match = re.search(total_pattern, sequence)
        if total_match:
            result['total'] = total_match.group(1)

        # Extract subtotal
        subtotal_pattern = r'<s_sub_total>.*?<s_subtotal_price>(.*?)</s_subtotal_price>'
        subtotal_match = re.search(subtotal_pattern, sequence)
        if subtotal_match:
            result['subtotal'] = subtotal_match.group(1)

        return result

    def _extract_fields_from_parsed(
        self,
        parsed_data: Dict[str, Any],
        raw_text: str
    ) -> Dict[str, Any]:
        """Extract fields from parsed Donut output."""
        import re

        fields = {
            'date': None,
            'time': None,
            'total': None,
            'merchant': None,
            'date_confidence': 0.0,
            'total_confidence': 0.0,
            'merchant_confidence': 0.0
        }

        # Extract total from parsed data
        if parsed_data.get('total'):
            try:
                total_str = parsed_data['total'].replace(',', '.').replace(' ', '')
                # Remove currency symbols
                total_str = re.sub(r'[€$£₺]', '', total_str)
                fields['total'] = float(total_str)
                fields['total_confidence'] = 0.85
            except ValueError:
                pass

        # Try to extract from raw text if not found
        if not fields['total']:
            total_patterns = [
                r'(\d+[.,]\d{2})',
            ]
            for pattern in total_patterns:
                matches = re.findall(pattern, raw_text)
                if matches:
                    try:
                        # Take the last (usually total)
                        fields['total'] = float(matches[-1].replace(',', '.'))
                        fields['total_confidence'] = 0.6
                        break
                    except ValueError:
                        continue

        # Extract date from raw text
        date_patterns = [
            r'(\d{2}[/.-]\d{2}[/.-]\d{4})',
            r'(\d{2}[/.-]\d{2}[/.-]\d{2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, raw_text)
            if match:
                fields['date'] = match.group(1)
                fields['date_confidence'] = 0.7
                break

        return fields
