# Reference: Colab for logic and training model
"""
Validate generated QA pairs and captions against grader data.
Run on Colab where data/valid/ exists:
    python -m homework.validate_data
"""
import json
from collections import Counter
from pathlib import Path

from .generate_captions import generate_caption
from .generate_qa import generate_qa_pairs


def validate_qa(data_dir: str = "data"):
    """Validate QA pairs against valid_grader/balanced_qa_pairs.json"""
    data_dir = Path(data_dir)
    grader_file = data_dir / "valid_grader" / "balanced_qa_pairs.json"
    valid_dir = data_dir / "valid"

    if not grader_file.exists():
        print(f"Grader file not found: {grader_file}")
        return
    if not valid_dir.exists():
        print(f"Valid directory not found: {valid_dir}")
        return

    with open(grader_file) as f:
        grader_qa = json.load(f)

    # Group grader QA by image
    grader_by_image = {}
    for qa in grader_qa:
        img = qa["image_file"]
        if img not in grader_by_image:
            grader_by_image[img] = []
        grader_by_image[img].append(qa)

    total = 0
    matched = 0
    mismatches_by_type = Counter()
    mismatch_examples = []

    for info_file in sorted(valid_dir.glob("*_info.json")):
        base_name = info_file.stem.replace("_info", "")
        for vi in range(10):
            image_file = f"valid/{base_name}_{vi:02d}_im.jpg"
            image_path = valid_dir / f"{base_name}_{vi:02d}_im.jpg"
            if not image_path.exists():
                continue
            gqas = grader_by_image.get(image_file, [])
            if not gqas:
                continue
            our_qas = generate_qa_pairs(str(info_file), vi)
            our_qa_dict = {q["question"]: q["answer"] for q in our_qas}
            for gqa in gqas:
                total += 1
                q = gqa["question"]
                expected = gqa["answer"]
                ours = our_qa_dict.get(q, "MISSING")
                if ours.lower() == expected.lower():
                    matched += 1
                else:
                    # Classify mismatch
                    ql = q.lower()
                    if "what kart is the ego" in ql:
                        cat = "ego"
                    elif "what track" in ql:
                        cat = "track"
                    elif "how many" in ql:
                        cat = "count"
                    elif "left or right" in ql or "front of or behind" in ql:
                        cat = "spatial_dir"
                    elif "where is" in ql:
                        cat = "spatial_rel"
                    else:
                        cat = "other"
                    mismatches_by_type[cat] += 1
                    if len(mismatch_examples) < 30:
                        mismatch_examples.append(
                            f"  Q: {q}\n  Expected: {expected}\n  Ours: {ours}\n  Image: {image_file}"
                        )

    print(f"\n=== QA Validation ===")
    print(f"Match rate: {matched}/{total} ({matched/total*100:.1f}%)")
    if mismatches_by_type:
        print(f"\nMismatches by type:")
        for cat, count in mismatches_by_type.most_common():
            print(f"  {cat}: {count}")
        print(f"\nMismatch examples:")
        for m in mismatch_examples:
            print(m)
            print()


def validate_captions(data_dir: str = "data"):
    """Validate captions against valid_grader/all_mc_qas.json"""
    data_dir = Path(data_dir)
    grader_file = data_dir / "valid_grader" / "all_mc_qas.json"
    valid_dir = data_dir / "valid"

    if not grader_file.exists():
        print(f"Grader file not found: {grader_file}")
        return
    if not valid_dir.exists():
        print(f"Valid directory not found: {valid_dir}")
        return

    with open(grader_file) as f:
        grader_mc = json.load(f)

    total = 0
    matched = 0
    mismatch_examples = []

    for mc in grader_mc:
        image_file = mc["image_file"]
        correct_caption = mc["candidates"][mc["correct_index"]]

        # Parse image file to get info file and view index
        parts = Path(image_file).stem.split("_")
        base_name = parts[0]
        vi = int(parts[1])
        info_file = valid_dir / f"{base_name}_info.json"

        if not info_file.exists():
            continue

        our_captions = generate_caption(str(info_file), vi)
        total += 1

        # Check if correct caption is among our generated captions
        our_lower = [c.lower() for c in our_captions]
        if correct_caption.lower() in our_lower:
            matched += 1
        else:
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    f"  Expected: {correct_caption}\n  Ours: {our_captions}\n  Image: {image_file}"
                )

    print(f"\n=== Caption Validation ===")
    print(f"Match rate: {matched}/{total} ({matched/total*100:.1f}%)")
    if mismatch_examples:
        print(f"\nMismatch examples:")
        for m in mismatch_examples:
            print(m)
            print()


def main():
    import fire
    fire.Fire({"qa": validate_qa, "captions": validate_captions, "all": lambda data_dir="data": (validate_qa(data_dir), validate_captions(data_dir))})


if __name__ == "__main__":
    main()
