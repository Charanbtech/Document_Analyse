"""
predict_cli.py — Command-line interface for document classification.

Usage:
    python predict_cli.py --text "Your text here"
    python predict_cli.py --file path/to/document.pdf
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Document Classification System — CLI Predictor"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help="Raw text to classify")
    group.add_argument('--file', type=str, help="Path to .txt, .pdf, or .docx file")
    args = parser.parse_args()

    from models.classifier import predict, load_model
    from utils.preprocessor import extract_text_from_file

    print("\n📄 Document Classification System")
    print("=" * 45)

    try:
        model, class_names = load_model()
    except FileNotFoundError:
        print("❌  Model not found. Run training first:\n    python models/classifier.py")
        sys.exit(1)

    if args.text:
        text   = args.text
        source = "text"
        print(f"Input : (direct text, {len(text)} chars)")
    else:
        print(f"File  : {args.file}")
        text   = extract_text_from_file(args.file)
        source = "file"
        print(f"Chars extracted : {len(text)}")

    result = predict(text, model=model, class_names=class_names)

    print(f"\n✅ Predicted Category : {result['predicted_class']}")
    print(f"   Confidence        : {result['confidence']:.2%}")
    print("\nTop-5 predictions:")
    for p in result['top_predictions']:
        bar = '█' * int(p['probability'] * 30)
        print(f"  {p['class']:<35} {bar} {p['probability']:.2%}")
    print()


if __name__ == "__main__":
    main()
