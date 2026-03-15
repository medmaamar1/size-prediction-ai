# BodyM Training Pipeline

Train XGBoost body measurement models on the BodyM dataset using
MediaPipe Pose for feature extraction.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Download BodyM Dataset

```bash
# Install AWS CLI if needed
pip install awscli

# Download (free, no AWS account needed for open data)
aws s3 cp s3://bodym-dataset/ ./bodym/ --recursive --no-sign-request
```

Expected structure after download:
```
bodym/
    train/
        images/          ← silhouette images
        train_labels.csv
    test_a/
        images/
        test_a_labels.csv
    test_b/
        images/
        test_b_labels.csv
```

---

## Train

```bash
# Full training run
python train.py --data_dir ./bodym --output_dir ./models

# Quick test run (first 200 rows only)
python train.py --data_dir ./bodym --output_dir ./models --max_rows 200

# Skip re-running MediaPipe if features already cached
python train.py --data_dir ./bodym --output_dir ./models --skip_images
```

Training output:
```
Training MALE model — 1,847 subjects
  chest_cm         CV MAE:  3.21 cm
  waist_cm         CV MAE:  3.84 cm
  hip_cm           CV MAE:  3.12 cm
  shoulder_width_cm CV MAE: 2.67 cm
  arm_length_cm    CV MAE:  2.11 cm
  leg_length_cm    CV MAE:  2.43 cm

Training FEMALE model — 658 subjects
  chest_cm         CV MAE:  4.12 cm
  waist_cm         CV MAE:  3.91 cm
  hip_cm           CV MAE:  3.44 cm
  ...

EVALUATION — Test Set A (lab)
  chest_cm         MAE:  3.8 cm   < 5cm: 78.2%   < 3cm: 51.4%
  waist_cm         MAE:  4.1 cm   < 5cm: 74.1%   < 3cm: 48.2%
  hip_cm           MAE:  3.5 cm   < 5cm: 80.3%   < 3cm: 55.1%

EVALUATION — Test Set B (in-the-wild)
  chest_cm         MAE:  5.2 cm   ...
  waist_cm         MAE:  5.8 cm   ...
```

---

## Predict on New Photos

```python
from predict import MeasurementPredictor

predictor = MeasurementPredictor('./models')

result = predictor.predict(
    front_image_path = 'user_front.jpg',
    side_image_path  = 'user_side.jpg',   # optional, improves circumferences
    height_cm        = 175,
    weight_kg        = 75,
    gender           = 'male',
    body_type        = 'muscular',         # see options below
    muscle_level     = 2,                  # 0=low 1=mid 2=high
    fat_level        = 0,
)

print(result)
# {
#   'chest_cm': 98.4,
#   'waist_cm': 83.2,
#   'hip_cm':   95.1,
#   'shoulder_width_cm': 46.2,
#   'arm_length_cm': 61.3,
#   'leg_length_cm': 97.8,
#   'fat_score': 1,
#   'muscle_score': 6,
#   'skinny_fat_score': 0,
#   'accuracy_mode': 'front+side (high)',
#   'warnings': ['Muscular build — standard size fits waist but may be tight at shoulders.']
# }
```

### Body type options

**Male:** `skinny` · `average` · `muscular` · `fat` · `skinny_fat`

**Female:** `slim` · `curvy` · `pear` · `apple` · `athletic` · `full_bust`

For women, also pass `bra_size='36C'` for best bust accuracy.

---

## Models saved

```
models/
    male_chest_cm.json
    male_waist_cm.json
    male_hip_cm.json
    male_shoulder_width_cm.json
    male_arm_length_cm.json
    male_leg_length_cm.json
    male_feature_cols.json       ← feature list for inference
    female_chest_cm.json
    female_waist_cm.json
    ...
    training_summary.json        ← MAE per model
```

---

## Photo requirements for best accuracy

- Stand straight, arms ~30° out from body
- Tight clothing (t-shirt + leggings/shorts)
- Full body visible — head to feet
- Good lighting, plain background preferred
- Camera at hip height, ~1.5–2.5m distance
- Side photo: turn 90°, same distance and pose
