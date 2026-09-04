# Adaptive Color Grading
## Trevor D. Canham, Abhijith Punnappurath, and Michael S. Brown

![](https://github.com/SamsungLabs/adaptive-color-grading/blob/main/dbAdaCGUI.png)
This is the open source repository for **Adaptive Color Grading** which will be presented at the 34th Color and Imaging Conference.
The tool consists of the following key classes:

`colorEngine` splits an image into four tonescale regions — `darkest`, `dark`, `light`,
`lightest` — each with a **tonescale region threshold (TRT)** (where the region begins, in relative luminance) and a
**falloff** (how quickly it blends out). An independent CIELAB a\*b\*
color offset can be applied to each region. The four region adjustments are composed into a single 3D LUT and applied to
the image, as in standard color grading modules.

The tool is novel in that it learns the *pivots* from the image itself: `adapt` fits
a k-nearest-neighbour regressor mapping a luminance histogram to the four pivot positions,
so a grade hand-authored on a small curated set can be pushed across a whole project as a
batch edit.

Users can interact with these classes via the nested color grading interface `adaCG.py` and database manager interface `dBadaCG.py`. The package also contains an example training dataset from the experiments in the paper, and a number of demo images for users to experiment with.

## Requirements

Python 3.12. Direct runtime dependencies:

- [`colour-science`](https://www.colour-science.org/) — color space conversion and 3D LUTs
- `numpy`
- `scikit-learn` — `KNeighborsRegressor`, `StandardScaler`
- `Pillow` — image resizing, thumbnail JPEG encode/decode
- `matplotlib` — the histogram and 3D LUT panes in `dBadaCG`
- `tkinter` — the UI. Ships with the Python install itself and is not pip-installable. If
  `import tkinter` fails, the interpreter was built without Tk.

## Install

From your preferred python environment (e.g., miniconda):

```bash
conda create -n acg python=3.12
```

```bash
conda activate acg
```

```bash
pip install -r requirements.txt
```

## Run
For the grading interface with database manager:

```bash
python dBadaCG.py
```

For the grading interface alone:

```bash
python adaCG.py
```

See classes colorEngine and adapt in adaCG.py for the color grading engine and adaptation model alone, respectively.

## Workflow

1. **Import.** The Project pane's `import` button adds a directory of images, computing a
   luminance histogram and a thumbnail for each.
2. **Grade.** Select a grade to load it into the editor. Drag on the chroma box to set the
   a\*b\* offset for the region chosen in the dropdown; drag on the tonescale ramp to move
   that region's pivot. Set `color offset applies to:` to `batch` to drive the whole Project
   selection at once.
3. **Curate.** `add`, `remove`, `label`, `filter` and visualize training set examples and their grades.
4. **Apply.** `apply model to selected` fits the KNN on the Train pane's *filtered* grades
   and writes predicted pivots onto the *selected* Project grades. Only the four pivots
   change; a\*b\* offsets and falloffs are left alone.
5. **Save.** `save project` writes the Project `gradeDict`; `save model` writes the Train
   `gradeDict`.

Nothing on disk is written except by `Save Project`, `Save Project As`, `Save Model`,
`Save Model As`, and the image/LUT export commands. Unsaved edits are marked with `*` in the
title bar and confirmed on quit.

## The `gradeDict`

Every component (`dBadaCG`, `adaCG`, `colorEngine`, `adapt`) communicates through the project dictionary.

```python
{
  'root': 'C:/path/to/images',          # where the images live
  'IMG_0001.tif': {                     # key: path relative to root
      'set':    {'darkest': [a, b], 'dark': [a, b], 'light': [a, b], 'lightest': [a, b]},
      'reg':    {'darkest': [pivot, falloff], ...},
      'hist':   np.ndarray,             # luminance histogram, the adapt input feature
      'thumb':  np.ndarray,             # JPEG byte stream, decoded on demand
      'labels': ['subsetA', ...],       # free-form tags, filterable
      'time':   datetime,
  },
  ...
}
```

Project and Train files use the same layout, so any project pickle can be imported as a
train set and vice versa.

## Settings

Under the `Settings` menu:

- **histogram bit depth** — bin count for the `adapt` input feature. Default 8-bit (256).
- **histogram handling** — `none`, `unit`, `std` (default), `log`, `cdf`.
- **Rehistogram Project** — recomputes every project histogram at the current settings.
  Needed after a bit depth change, since `adapt` refuses to mix bin counts.
- **adapt neighbors** — *k* for the KNN. Default 16.

Preview resolution is fixed at one sixth of the screen width, set by `previewDiv` in
`dBadaCGapp.__init__`. It is not exposed in the menu because changing it forces a full
re-read of the project; the menu entry is left commented out next to the `Settings` build.

## How to cite
```
@inproceedings{acg,
  title={Adaptive Color Grading},
  author={Canham, Trevor and Punnappurath, Abhijith and Brown, Michael},
  booktitle={Color and Imaging Conference},
  year={2026}
}
```
