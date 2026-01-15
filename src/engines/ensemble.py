"""
Ensemble OCR Manager
Runs multiple OCR engines and compares results.
"""

from typing import List, Dict, Any, Optional, Type
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from .base import BaseOCREngine, OCRResult
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .donut_engine import DonutEngine
from .got_ocr_engine import GOTOCREngine

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing multiple OCR engines."""

    image_path: str
    results: Dict[str, OCRResult] = field(default_factory=dict)
    best_engine: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_field_comparison(self, field_name: str) -> Dict[str, Any]:
        """Get comparison for a specific field across engines."""
        comparison = {}
        for engine_name, result in self.results.items():
            value = result.fields.get(field_name)
            confidence = result.fields.get(f'{field_name}_confidence', 0.0)
            comparison[engine_name] = {
                'value': value,
                'confidence': confidence
            }
        return comparison


class EnsembleOCR:
    """
    Ensemble OCR manager for running and comparing multiple OCR engines.
    """

    # Available engine classes
    ENGINE_CLASSES: Dict[str, Type[BaseOCREngine]] = {
        'easyocr': EasyOCREngine,
        'paddleocr': PaddleOCREngine,
        'donut': DonutEngine,
        'got-ocr': GOTOCREngine
    }

    def __init__(self, engines: List[str] = None, preprocess: bool = True):
        """
        Initialize ensemble manager.

        Args:
            engines: List of engine names to use.
                    If None, uses all available engines.
            preprocess: Apply document scanning preprocessing (default: True)
        """
        self.engines: Dict[str, BaseOCREngine] = {}
        self.preprocess = preprocess
        self._initialize_engines(engines)

    def _initialize_engines(self, engine_names: List[str] = None) -> None:
        """Initialize requested engines that are available."""
        if engine_names is None:
            engine_names = list(self.ENGINE_CLASSES.keys())

        for name in engine_names:
            if name not in self.ENGINE_CLASSES:
                logger.warning(f"Unknown engine: {name}")
                continue

            engine_class = self.ENGINE_CLASSES[name]

            if engine_class.is_available():
                try:
                    self.engines[name] = engine_class(preprocess=self.preprocess)
                    logger.info(f"Engine registered: {name}")
                except Exception as e:
                    logger.error(f"Failed to create engine {name}: {e}")
            else:
                logger.warning(f"Engine not available (missing dependencies): {name}")

    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return list(self.engines.keys())

    def process(
        self,
        image_path: str,
        engines: List[str] = None
    ) -> ComparisonResult:
        """
        Process image with multiple engines and compare results.

        Args:
            image_path: Path to image file
            engines: Specific engines to use (None = all available)

        Returns:
            ComparisonResult with results from all engines
        """
        comparison = ComparisonResult(image_path=image_path)

        # Determine which engines to use
        engine_names = engines or list(self.engines.keys())

        for name in engine_names:
            if name not in self.engines:
                logger.warning(f"Engine not available: {name}")
                continue

            engine = self.engines[name]

            try:
                logger.info(f"Processing with {name}...")
                result = engine.extract(image_path)
                comparison.results[name] = result
                logger.info(f"  {name}: confidence={result.confidence:.2%}, time={result.processing_time:.2f}s")
            except Exception as e:
                logger.error(f"Engine {name} failed: {e}")
                # Create empty result for failed engine
                comparison.results[name] = OCRResult(
                    engine_name=name,
                    metadata={'error': str(e)}
                )

        # Determine best engine based on confidence
        if comparison.results:
            best_name = max(
                comparison.results.keys(),
                key=lambda k: comparison.results[k].confidence
            )
            comparison.best_engine = best_name

        # Generate summary
        comparison.summary = self._generate_summary(comparison)

        return comparison

    def _generate_summary(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate summary statistics for comparison."""
        summary = {
            'total_engines': len(comparison.results),
            'successful_engines': sum(
                1 for r in comparison.results.values()
                if r.confidence > 0
            ),
            'best_engine': comparison.best_engine,
            'field_winners': {}
        }

        # Determine best engine for each field
        fields = ['date', 'total', 'merchant']
        for field in fields:
            best_engine = None
            best_confidence = 0.0

            for engine_name, result in comparison.results.items():
                conf = result.fields.get(f'{field}_confidence', 0.0)
                if conf > best_confidence:
                    best_confidence = conf
                    best_engine = engine_name

            summary['field_winners'][field] = {
                'engine': best_engine,
                'confidence': best_confidence
            }

        return summary

    def print_comparison_table(
        self,
        comparison: ComparisonResult,
        show_raw_text: bool = False
    ) -> None:
        """
        Print formatted comparison table to terminal.

        Args:
            comparison: ComparisonResult to display
            show_raw_text: Whether to show raw OCR text
        """
        print("\n" + "=" * 80)
        print(f"OCR COMPARISON RESULTS")
        print(f"Image: {comparison.image_path}")
        print("=" * 80)

        # Engine summary table
        print("\n+" + "-" * 78 + "+")
        print(f"| {'ENGINE':<15} | {'CONFIDENCE':>12} | {'TIME (s)':>10} | {'STATUS':<30} |")
        print("+" + "-" * 78 + "+")

        for name, result in comparison.results.items():
            status = "[OK]" if result.confidence > 0 else "[X] Failed"
            if result.metadata.get('error'):
                status = f"[X] {result.metadata['error'][:25]}"

            conf_str = f"{result.confidence:.1%}" if result.confidence > 0 else "N/A"
            time_str = f"{result.processing_time:.2f}" if result.processing_time > 0 else "N/A"

            marker = "*" if name == comparison.best_engine else " "
            print(f"|{marker}{name:<14} | {conf_str:>12} | {time_str:>10} | {status:<30} |")

        print("+" + "-" * 78 + "+")
        print("  * = Best overall confidence")

        # Field comparison table
        print("\n+" + "-" * 78 + "+")
        print(f"| {'FIELD':<12} |", end="")
        for name in comparison.results.keys():
            print(f" {name:<18} |", end="")
        print()
        print("+" + "-" * 78 + "+")

        fields = [
            ('merchant', 'Merchant'),
            ('date', 'Date'),
            ('time', 'Time'),
            ('total', 'Total'),
        ]

        for field_key, field_name in fields:
            print(f"| {field_name:<12} |", end="")

            best_conf = 0
            best_engine = None
            for name, result in comparison.results.items():
                conf = result.fields.get(f'{field_key}_confidence', 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_engine = name

            for name, result in comparison.results.items():
                value = result.fields.get(field_key)
                conf = result.fields.get(f'{field_key}_confidence', 0.0)

                if value is None:
                    display = "-"
                elif isinstance(value, float):
                    display = f"{value:.2f}"
                else:
                    display = str(value)[:15]

                # Truncate if too long
                if len(display) > 15:
                    display = display[:12] + "..."

                # Mark best with star
                marker = "*" if name == best_engine and best_conf > 0 else " "
                print(f"{marker}{display:<17} |", end="")

            print()

        print("+" + "-" * 78 + "+")
        print("  * = Highest confidence for this field")

        # Confidence details
        print("\n+" + "-" * 78 + "+")
        print(f"| {'CONFIDENCE':<12} |", end="")
        for name in comparison.results.keys():
            print(f" {name:<18} |", end="")
        print()
        print("+" + "-" * 78 + "+")

        for field_key, field_name in fields:
            print(f"| {field_name:<12} |", end="")
            for name, result in comparison.results.items():
                conf = result.fields.get(f'{field_key}_confidence', 0.0)
                conf_str = f"{conf:.1%}" if conf > 0 else "-"
                print(f" {conf_str:<17} |", end="")
            print()

        print("+" + "-" * 78 + "+")

        # Summary
        print(f"\n## SUMMARY:")
        print(f"   Best Overall Engine: {comparison.best_engine}")
        print(f"   Field Winners:")
        for field, info in comparison.summary.get('field_winners', {}).items():
            if info['engine']:
                print(f"     - {field.capitalize()}: {info['engine']} ({info['confidence']:.1%})")

        # Raw text (optional)
        if show_raw_text:
            print("\n" + "=" * 80)
            print("RAW OCR TEXT")
            print("=" * 80)
            for name, result in comparison.results.items():
                print(f"\n--- {name} ---")
                print(result.raw_text[:500] if result.raw_text else "(empty)")
                if len(result.raw_text) > 500:
                    print("... (truncated)")

        print()

    def to_json(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Convert comparison result to JSON-serializable dict."""
        return {
            'image_path': comparison.image_path,
            'best_engine': comparison.best_engine,
            'summary': comparison.summary,
            'results': {
                name: result.to_dict()
                for name, result in comparison.results.items()
            }
        }


def compare_engines(
    image_path: str,
    engines: List[str] = None,
    show_table: bool = True,
    preprocess: bool = True
) -> ComparisonResult:
    """
    Convenience function to compare OCR engines on an image.

    Args:
        image_path: Path to image file
        engines: List of engine names (None = all available)
        show_table: Print comparison table to terminal
        preprocess: Apply document scanning preprocessing (default: True)

    Returns:
        ComparisonResult with all results
    """
    ensemble = EnsembleOCR(engines=engines, preprocess=preprocess)
    result = ensemble.process(image_path, engines=engines)

    if show_table:
        ensemble.print_comparison_table(result)

    return result
