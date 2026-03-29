import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    with open(info_path) as f:
        info = json.load(f)

    captions = []

    # find ego kart (track_id = 0)
    ego_kart = None
    other_karts = []
    for k in karts:
        if k["instance_id"] == 0:
            ego_kart = k
        else:
            other_karts.append(k)

    if ego_kart is None:
        return captions

    ego_cx = ego_kart["center"][0]
    ego_distance = info["distance_down_track"][0]

    # 1. Ego car
    # {kart_name} is the ego car.
    captions.append(f"{ego_kart['kart_name']} is the ego car.")

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(f"There are {len(karts)} karts in the scenario.")

    # 3. Track name
    # The track is {track_name}.
    captions.append(f"The track is {track_name}.")

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    for k in other_karts:
        kart_name = k["kart_name"]
        kart_cx = k["center"][0]
        kart_distance = info["distance_down_track"][k["instance_id"]]

        if kart_cx < ego_cx:
            lr = "left"
        else:
            lr = "right"

        if kart_distance > ego_distance:
            fb = "front"
        else:
            fb = "back"

        captions.append(f"{kart_name} is {fb} and {lr} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def generate_all(data_dir: str = "data/train", output_file: str = "data/train/generated_captions.json"):
    info_files = sorted(Path(data_dir).glob("*_info.json"))
    all_captions = []
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = f"train/{base_name}_{view_index:02d}_im.jpg"
            image_path = Path(data_dir) / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_path.exists():
                continue
            captions = generate_caption(str(info_file), view_index)
            for cap in captions:
                all_captions.append({
                    "image_file": image_file,
                    "caption": cap,
                })
    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=2)
    print(f"Generated {len(all_captions)} captions, saved to {output_file}")


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()
