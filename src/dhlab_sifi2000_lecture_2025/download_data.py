from pathlib import Path

import datasets
import PIL.Image as Image


def save_image_and_transcription(sample: dict, parent: Path, idx: int) -> None:
    img: Image.Image = sample["image"]
    text: str = sample["text"]
    img.save(parent / f"{idx:02d}.png")
    (parent / f"{idx:02d}.txt").write_text(text)


dataset = datasets.load_dataset("Teklia/NorHand-v3-line")
val_data = dataset["validation"]

data_dir = Path(__file__).parent.parent.parent / "notebooks/data"
data_dir.mkdir(exist_ok=True)

# Get first image
save_image_and_transcription(val_data[101], data_dir, 0)
save_image_and_transcription(val_data[0], data_dir, 1)
save_image_and_transcription(
    val_data.filter(lambda row: row["text"] == "kan ein ikkje her. Aagot var lukkeleg")[0],
    data_dir,
    2
)

(data_dir / "README.md").write_text(
    """\
# HTR data from the NorHand-v3-line dataset.

These images and transcriptions are taken from the validation set of the [NorHand-v3-line dataset](https://huggingface.co/datasets/Teklia/NorHand-v3-line).
This dataset is a transformed version of the NorHand v3 dataset published on Zenodo under a CC-BY 4.0 license: https://zenodo.org/records/10255840

CC-BY 4.0

Beyer, Y., & Solberg, P. E. (2023). NorHand v3 / Dataset for Handwritten Text Recognition in Norwegian [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10255840
"""
)
