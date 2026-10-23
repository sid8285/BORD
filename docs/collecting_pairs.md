# Collecting the Checkpoint 2 dataset

## Honest pairs (15 for CP2)

1. Pick tasks with a visible state change: clean desk, made bed, empty sink, folded laundry, cleared floor. Repeat **some tasks in the same spot on different days**. The repeats are what let the script build "reused" attacks.
2. For each attempt, take a **before** photo, do the task (or deliberately don't, for a few `not_completed` pairs), and then take an **after** photo from roughly the same angle.
3. Use JPEG. On iPhone, set Settings > Camera > Formats > Most Compatible, or export as JPEG. HEIC files will not load.
4. Save the photos as `data/pairs/<pair_id>/before.jpg` and `after.jpg`, with IDs `p001`, `p002`, and so on.
5. Add one row per pair to `data/pairs.csv`:

```csv
pair_id,task_type,task_description,label,scene_id
p001,single_scene,Clean my desk,completed,desk
p002,single_scene,Make my bed,not_completed,bed
```

A reasonable CP2 mix is 15 pairs across 4 or 5 scenes, about 12 `completed` and 3 `not_completed`, with each scene photographed on at least 2 days.

## Adversarial set

The **automatic** categories come from your honest pairs, with nothing extra to photograph:
- `reused`: a later attempt submitted with an earlier attempt's after photo from the same scene (any task; kept for comparison with Checkpoint 2)
- `resubmitted`: a later attempt at the **same task** submitted with that task's earlier after photo. It needs a `task_id` column in `pairs.csv`: give the same id to every pair that is the same task done on a different day, for example

```csv
pair_id,task_type,task_description,label,scene_id,task_id
p016,single_scene,Close the laptop,completed,desk,laptop
p024,single_scene,Close the laptop,completed,desk,laptop
```
- `wrong_scene`: a before photo paired with an after photo from a different scene

The **semi-automatic** category needs downloaded images:
- `web_sourced`: put 5 to 15 stock or web images of clean rooms in `data/web/`, for example from the Kaggle Messy vs Clean Room set in `dataset_spec.md`. Check the image license before committing.

The **by hand** categories are optional, but they add the categories the baseline is most likely to miss:
- `staged_partial`: clean only the photographed corner, then save it as `data/adversarial/<pair_id>-staged_partial.jpg`.
- `ai_edited`: edit the real before photo with an image model so it looks clean, then save it as `data/adversarial/<pair_id>-ai_edited.jpg`.
- `recaptured`: for a repeated task, open the **earlier** attempt's after photo full screen on a laptop or another phone, photograph the screen with the camera, and save it as `data/adversarial/<pair_id>-recaptured.jpg`, where `<pair_id>` is the **later** attempt. Vary it: fill the frame on some, show the screen edge on others.

Then build and check the manifest:

```bash
cd backend
python -m bord.build_manifest --data ../data
```

It prints the honest and adversarial counts per category. It also warns about pairs whose EXIF capture times are missing or out of order.

**Before committing photos, strip the GPS location.** Phone photos usually record where they were taken, and the repository is public. `build_manifest` warns when it finds GPS data. [exiftool](https://exiftool.org) (`brew install exiftool`) removes it without re-encoding the image, and it keeps the capture time that the timestamp check uses:

```bash
exiftool -r -gps:all= -overwrite_original data/   # -r reaches the photos in data/pairs/<id>/
exiftool -r -gps:all data/                          # check: should list file names but no GPS tags
```
