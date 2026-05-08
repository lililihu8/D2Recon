import argparse
import os
import os.path as osp


def readlines(path):
    with open(path, "r") as fh:
        return [line.strip() for line in fh.readlines() if line.strip()]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_scene_name(scene_name):
    suffix = "_with_camera_labels"
    return scene_name[:-len(suffix)] if scene_name.endswith(suffix) else scene_name


def collect_available_frames(output_root, split_name, folder_name, img_ext):
    rgb_dir = osp.join(output_root, split_name, folder_name, "videos", "full_set_30000", "rgbs")
    if not osp.isdir(rgb_dir):
        return set()
    frames = set()
    for fname in os.listdir(rgb_dir):
        stem, ext = osp.splitext(fname)
        if ext != img_ext:
            continue
        frames.add(int(stem.split("_")[0]))
    return frames


def remap_split_lines(source_lines, scene_names, split_prefix, output_root, img_ext):
    remapped = []
    scene_to_folder = {normalize_scene_name(scene): f"{idx:03d}" for idx, scene in enumerate(scene_names)}
    available_frames = {
        scene: collect_available_frames(output_root, split_prefix, folder_name, img_ext)
        for scene, folder_name in scene_to_folder.items()
    }

    for line in source_lines:
        folder, frame_idx = line.split()[:2]
        scene_name = normalize_scene_name(folder.split("/", 1)[1])
        if scene_name not in scene_to_folder:
            continue

        frame_idx = int(frame_idx)
        if not available_frames[scene_name] or frame_idx not in available_frames[scene_name]:
            continue

        remapped.append(f"{split_prefix}/{scene_to_folder[scene_name]} {frame_idx}")

    return remapped


def main():
    parser = argparse.ArgumentParser(description="Generate render dataset split files from selected Waymo scenes.")
    repo_root = osp.dirname(osp.dirname(__file__))
    parser.add_argument("--repo_root", default=repo_root)
    parser.add_argument("--output_root", default="/mnt/hdd_14/Data_for_Dynamo/Output")
    parser.add_argument("--train_scene_split", default=osp.join(repo_root, "train_split.txt"))
    parser.add_argument("--val_scene_split", default=osp.join(repo_root, "val_split.txt"))
    parser.add_argument("--waymo_train_files", default=osp.join(repo_root, "splits", "waymo", "train_files.txt"))
    parser.add_argument("--waymo_test_files", default=osp.join(repo_root, "splits", "waymo", "test_files.txt"))
    parser.add_argument("--out_split_dir", default=osp.join(repo_root, "splits", "render"))
    parser.add_argument("--img_ext", default=".jpg", choices=[".jpg", ".png"])
    args = parser.parse_args()

    ensure_dir(args.out_split_dir)

    train_scene_names = readlines(args.train_scene_split)
    val_scene_names = readlines(args.val_scene_split)

    train_source = readlines(args.waymo_train_files)
    test_source = readlines(args.waymo_test_files)

    train_files = remap_split_lines(train_source, train_scene_names, "train", args.output_root, args.img_ext)
    val_files = remap_split_lines(test_source, val_scene_names, "val", args.output_root, args.img_ext)

    outputs = {
        "train_files.txt": train_files,
        "val_files.txt": val_files,
        "test_files.txt": val_files,
        "test_mask_files.txt": [],
    }

    for filename, lines in outputs.items():
        with open(osp.join(args.out_split_dir, filename), "w") as fh:
            for line in lines:
                fh.write(line + "\n")

    print(f"Wrote render split to {args.out_split_dir}")
    print(f"train_files: {len(train_files)}")
    print(f"val_files: {len(val_files)}")
    print("test_mask_files: 0 (render dataset has no segmentation annotations)")


if __name__ == "__main__":
    main()
