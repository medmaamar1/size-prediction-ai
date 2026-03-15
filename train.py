"""
BodyM Training Pipeline
=======================
Trains two XGBoost models (male + female) on the BodyM dataset.
Predicts: chest, waist, hip, shoulder_width, arm_length, leg_length (all in cm)

BodyM dataset structure expected:
    bodym/
        train/
            images/          ← front + side silhouettes (not needed for XGBoost)
            train_labels.csv ← ground truth measurements
        test_a/
            test_a_labels.csv
        test_b/
            test_b_labels.csv

Usage:
    python train.py --data_dir ./bodym --output_dir ./models
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import mediapipe as mp
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

mp_pose = mp.solutions.pose


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# MediaPipe landmark indices
LM = {
    'nose':         0,
    'left_eye':     2,
    'right_eye':    5,
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
    'left_knee':      25,
    'right_knee':     26,
    'left_ankle':     27,
    'right_ankle':    28,
    'left_heel':      29,
    'right_heel':     30,
}

# Torso fractions for mask scanning (0 = shoulder level, 1 = hip level)
MASK_LEVELS = {
    'armpit':  0.05,
    'chest':   0.20,
    'underbust': 0.35,
    'waist':   0.55,
    'navel':   0.65,
    'hip':     1.00,
    'thigh':   1.30,   # below hip line
}

# Target measurements to predict
TARGETS = [
    'chest_cm',
    'waist_cm',
    'hip_cm',
    'shoulder_width_cm',
    'arm_length_cm',
    'leg_length_cm',
]

# Body type thresholds (used to derive body_type feature)
BMI_BINS   = [0, 18.5, 25.0, 30.0, 100]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese']


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def px_dist(lm_a, lm_b, fw, fh):
    """Euclidean pixel distance between two landmarks."""
    ax, ay = lm_a.x * fw, lm_a.y * fh
    bx, by = lm_b.x * fw, lm_b.y * fh
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def compute_scale_factor(lm, fh, height_cm):
    """
    px_per_cm using eye-to-heel distance.
    Eye landmark is closer to crown than nose → more accurate.
    1.04 correction factor for remaining gap to true crown.
    """
    eye_y   = ((lm[LM['left_eye']].y + lm[LM['right_eye']].y) / 2) * fh
    heel_y  = ((lm[LM['left_heel']].y + lm[LM['right_heel']].y) / 2) * fh
    body_px = (heel_y - eye_y) * 1.04
    if body_px <= 0:
        return None
    return body_px / height_cm


def mask_width_at_fraction(mask, lm, fraction, fh, fw):
    """
    Scan the segmentation mask at a given fraction between
    shoulder line and hip line. Returns width in pixels.
    fraction < 0 → above shoulders, fraction > 1 → below hips.
    """
    sh_y  = ((lm[LM['left_shoulder']].y + lm[LM['right_shoulder']].y) / 2) * fh
    hip_y = ((lm[LM['left_hip']].y    + lm[LM['right_hip']].y)    / 2) * fh

    target_y = int(sh_y + (hip_y - sh_y) * fraction)
    target_y = max(0, min(target_y, fh - 1))

    row  = mask[target_y, :]
    cols = np.where(row > 0.5)[0]
    if len(cols) == 0:
        return 0, target_y
    return int(cols[-1] - cols[0]), target_y


def validate_landmarks(lm, threshold=0.65):
    """Check all critical landmarks are visible."""
    critical = [
        LM['left_shoulder'], LM['right_shoulder'],
        LM['left_hip'],      LM['right_hip'],
        LM['left_ankle'],    LM['right_ankle'],
    ]
    return all(lm[i].visibility > threshold for i in critical)


def extract_features_from_image(img_path, height_cm, is_side=False):
    """
    Run MediaPipe on one image and extract all geometric features.
    Returns a dict of features or None if pose not detected.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    fh, fw = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    lm   = results.pose_landmarks.landmark
    mask = results.segmentation_mask  # shape (fh, fw), values 0–1

    if not validate_landmarks(lm):
        return None

    ppc = compute_scale_factor(lm, fh, height_cm)
    if ppc is None or ppc <= 0:
        return None

    prefix = 'side_' if is_side else 'front_'
    feats  = {}

    # ── Landmark-based distances ────────────────────────────────────────
    feats[f'{prefix}shoulder_px']  = px_dist(lm[LM['left_shoulder']], lm[LM['right_shoulder']], fw, fh)
    feats[f'{prefix}hip_px']       = px_dist(lm[LM['left_hip']],      lm[LM['right_hip']],      fw, fh)
    feats[f'{prefix}upper_arm_px'] = px_dist(lm[LM['left_shoulder']], lm[LM['left_elbow']],     fw, fh)
    feats[f'{prefix}lower_arm_px'] = px_dist(lm[LM['left_elbow']],    lm[LM['left_wrist']],     fw, fh)
    feats[f'{prefix}upper_leg_px'] = px_dist(lm[LM['left_hip']],      lm[LM['left_knee']],      fw, fh)
    feats[f'{prefix}lower_leg_px'] = px_dist(lm[LM['left_knee']],     lm[LM['left_ankle']],     fw, fh)
    feats[f'{prefix}torso_px']     = px_dist(lm[LM['left_shoulder']], lm[LM['left_hip']],       fw, fh)

    # Convert to cm
    for k in list(feats.keys()):
        if k.endswith('_px'):
            feats[k.replace('_px', '_cm')] = feats[k] / ppc

    # ── Mask width scan at all levels ───────────────────────────────────
    for level_name, fraction in MASK_LEVELS.items():
        w_px, scan_y = mask_width_at_fraction(mask, lm, fraction, fh, fw)
        feats[f'{prefix}mask_{level_name}_px'] = w_px
        feats[f'{prefix}mask_{level_name}_cm'] = w_px / ppc if ppc > 0 else 0

    # ── Ratios (scale-invariant shape descriptors) ──────────────────────
    sh_px    = feats[f'{prefix}shoulder_px']
    hip_px   = feats[f'{prefix}hip_px']
    waist_px = feats[f'{prefix}mask_waist_px']
    chest_px = feats[f'{prefix}mask_chest_px']

    feats[f'{prefix}shoulder_waist_ratio'] = sh_px / waist_px  if waist_px > 0 else 0
    feats[f'{prefix}shoulder_hip_ratio']   = sh_px / hip_px    if hip_px   > 0 else 0
    feats[f'{prefix}waist_hip_ratio']      = waist_px / hip_px if hip_px   > 0 else 0
    feats[f'{prefix}chest_waist_ratio']    = chest_px / waist_px if waist_px > 0 else 0
    feats[f'{prefix}waist_chest_ratio']    = waist_px / chest_px if chest_px > 0 else 0

    # ── Visibility scores (confidence) ──────────────────────────────────
    feats[f'{prefix}vis_shoulder'] = (lm[LM['left_shoulder']].visibility + lm[LM['right_shoulder']].visibility) / 2
    feats[f'{prefix}vis_hip']      = (lm[LM['left_hip']].visibility      + lm[LM['right_hip']].visibility)      / 2
    feats[f'{prefix}vis_ankle']    = (lm[LM['left_ankle']].visibility     + lm[LM['right_ankle']].visibility)    / 2

    feats[f'{prefix}px_per_cm']    = ppc
    feats[f'{prefix}frame_h']      = fh
    feats[f'{prefix}frame_w']      = fw

    return feats


def derive_body_type_scores(row):
    """
    Rule-based body shape scores from measurements + ratios.
    These become additional features for the model.
    """
    bmi = row.get('bmi', 22)
    sh_w_ratio = row.get('front_shoulder_waist_ratio', 1.2)
    w_h_ratio  = row.get('front_waist_hip_ratio', 0.85)
    chest_waist = row.get('front_chest_waist_ratio', 1.1)

    # Fat score (0–9)
    fat = 0
    if bmi > 30:              fat += 3
    if sh_w_ratio < 1.15:     fat += 2
    if w_h_ratio > 0.90:      fat += 2
    if row.get('front_mask_waist_px', 0) > row.get('front_mask_chest_px', 1): fat += 2

    # Muscular score (0–8)
    muscle = 0
    if sh_w_ratio > 1.45:     muscle += 3
    if 24 < bmi < 30:         muscle += 1
    arm_cm  = row.get('front_upper_arm_cm', 0) + row.get('front_lower_arm_cm', 0)
    sh_cm   = row.get('front_shoulder_cm', 40)
    if arm_cm > sh_cm * 0.55: muscle += 2
    if sh_w_ratio > 1.35:     muscle += 2

    # Skinny-fat score (0–5)
    skinny_fat = 0
    if 18.5 < bmi < 25 and sh_w_ratio < 1.20: skinny_fat += 3
    if w_h_ratio > 0.85 and bmi < 25:         skinny_fat += 2

    return {'fat_score': fat, 'muscle_score': muscle, 'skinny_fat_score': skinny_fat}


# ═══════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_labels(csv_path):
    """
    Load BodyM label CSV.
    Expected columns (BodyM format):
        subject_id, gender, height_cm, weight_kg,
        chest_cm, waist_cm, hip_cm,
        bicep_cm, forearm_cm, wrist_cm,
        thigh_cm, calf_cm, ankle_cm,
        arm_length_cm, leg_length_cm, shoulder_width_cm,
        front_image, side_image
    """
    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase + underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Derive BMI
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2

    # Encode gender: male=0, female=1
    if 'gender' in df.columns:
        df['gender_enc'] = (df['gender'].str.lower().str.strip() == 'female').astype(int)

    # BMI bucket as ordinal
    df['bmi_bucket'] = pd.cut(df['bmi'], bins=BMI_BINS, labels=[0,1,2,3]).astype(float)

    print(f"Loaded {len(df)} subjects from {csv_path}")
    print(f"  Gender split: {df['gender'].value_counts().to_dict()}")
    print(f"  Missing targets: {df[TARGETS].isnull().sum().to_dict()}")
    return df


def build_feature_matrix(df, images_dir, max_rows=None):
    """
    Process all images through MediaPipe and build the full feature matrix.
    Skips rows where MediaPipe fails (returns None).
    """
    rows = []
    failed = 0
    total = len(df) if max_rows is None else min(max_rows, len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        if max_rows and i >= max_rows:
            break

        if i % 100 == 0:
            print(f"  Processing {i}/{total}  (failed: {failed})")

        height_cm = row['height_cm']

        # ── Front photo ────────────────────────────────────────────────
        front_path = os.path.join(images_dir, str(row.get('front_image', '')))
        front_feats = extract_features_from_image(front_path, height_cm, is_side=False)
        if front_feats is None:
            failed += 1
            continue

        # ── Side photo (optional — use if available) ───────────────────
        side_col = 'side_image' if 'side_image' in row.index else None
        side_feats = {}
        if side_col and pd.notna(row.get(side_col)):
            side_path = os.path.join(images_dir, str(row[side_col]))
            sf = extract_features_from_image(side_path, height_cm, is_side=True)
            if sf:
                side_feats = sf

        # ── Static inputs ──────────────────────────────────────────────
        static = {
            'height_cm':   height_cm,
            'weight_kg':   row.get('weight_kg', 70),
            'bmi':         row.get('bmi', 22),
            'bmi_bucket':  row.get('bmi_bucket', 1),
            'gender_enc':  row.get('gender_enc', 0),
            'has_side':    int(len(side_feats) > 0),
        }

        # ── Merge all features ─────────────────────────────────────────
        combined = {**static, **front_feats, **side_feats}

        # ── Body shape scores ──────────────────────────────────────────
        scores = derive_body_type_scores(combined)
        combined.update(scores)

        # ── Targets ───────────────────────────────────────────────────
        for t in TARGETS:
            combined[f'target_{t}'] = row.get(t, np.nan)

        combined['subject_id'] = row.get('subject_id', i)
        combined['gender']     = row.get('gender', 'unknown')

        rows.append(combined)

    df_feats = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(df_feats)} rows, {len(df_feats.columns)} columns")
    print(f"Dropped (MediaPipe fail): {failed} rows")
    return df_feats


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

XGBOOST_PARAMS = {
    'n_estimators':     500,
    'max_depth':        6,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'random_state':     42,
    'n_jobs':           -1,
    'tree_method':      'hist',   # fast on CPU
}


def get_feature_cols(df):
    """All columns that are not targets, IDs, or raw string fields."""
    exclude = {'subject_id', 'gender'} | {f'target_{t}' for t in TARGETS}
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
            and not c.startswith('target_')]


def train_gender_models(df_feats, output_dir):
    """
    Train separate male and female XGBoost models for each target measurement.
    Returns dict of models and evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    feature_cols = get_feature_cols(df_feats)

    print(f"\nFeatures used: {len(feature_cols)}")

    for gender in ['male', 'female']:
        gender_mask = df_feats['gender'].str.lower().str.strip().isin(
            ['male', 'm'] if gender == 'male' else ['female', 'f']
        )
        df_g = df_feats[gender_mask].copy()
        print(f"\n{'='*60}")
        print(f"Training {gender.upper()} model — {len(df_g)} subjects")
        print(f"{'='*60}")

        results[gender] = {}

        for target in TARGETS:
            target_col = f'target_{target}'
            if target_col not in df_g.columns:
                print(f"  Skipping {target} — not in labels")
                continue

            # Drop rows where target is missing
            df_t = df_g.dropna(subset=[target_col])
            if len(df_t) < 50:
                print(f"  Skipping {target} — only {len(df_t)} valid rows")
                continue

            X = df_t[feature_cols].fillna(0)
            y = df_t[target_col].values

            # ── 5-fold cross-validation ────────────────────────────────
            kf     = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_maes = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                m = xgb.XGBRegressor(**XGBOOST_PARAMS)
                m.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
                preds = m.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                fold_maes.append(mae)

            cv_mae = np.mean(fold_maes)
            print(f"  {target:<25} CV MAE: {cv_mae:.2f} cm  (folds: {[f'{m:.2f}' for m in fold_maes]})")

            # ── Train final model on all data ──────────────────────────
            final_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            final_model.fit(X, y, verbose=False)

            # Save model
            model_path = os.path.join(output_dir, f'{gender}_{target}.json')
            final_model.save_model(model_path)

            results[gender][target] = {
                'cv_mae_cm':   round(cv_mae, 3),
                'n_subjects':  len(df_t),
                'model_path':  model_path,
            }

        # Save feature column list (needed at inference time)
        feat_path = os.path.join(output_dir, f'{gender}_feature_cols.json')
        with open(feat_path, 'w') as f:
            json.dump(feature_cols, f)
        print(f"\n  Feature cols saved → {feat_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_on_test_set(df_test_feats, models_dir, set_name="Test"):
    """
    Load trained models and evaluate on a test feature matrix.
    Prints MAE per measurement and overall.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION — {set_name}")
    print(f"{'='*60}")

    all_errors = []

    for gender in ['male', 'female']:
        feat_path = os.path.join(models_dir, f'{gender}_feature_cols.json')
        if not os.path.exists(feat_path):
            continue

        with open(feat_path) as f:
            feature_cols = json.load(f)

        mask = df_test_feats['gender'].str.lower().str.strip().isin(
            ['male', 'm'] if gender == 'male' else ['female', 'f']
        )
        df_g = df_test_feats[mask]
        if len(df_g) == 0:
            continue

        print(f"\n{gender.upper()} ({len(df_g)} subjects)")
        print(f"  {'Measurement':<25} {'MAE (cm)':>10} {'< 5cm %':>10} {'< 3cm %':>10}")
        print(f"  {'-'*55}")

        for target in TARGETS:
            model_path = os.path.join(models_dir, f'{gender}_{target}.json')
            if not os.path.exists(model_path):
                continue
            target_col = f'target_{target}'
            if target_col not in df_g.columns:
                continue

            df_t = df_g.dropna(subset=[target_col])
            if len(df_t) == 0:
                continue

            X = df_t[feature_cols].fillna(0)
            y = df_t[target_col].values

            model = xgb.XGBRegressor()
            model.load_model(model_path)
            preds  = model.predict(X)
            errors = np.abs(preds - y)
            mae    = errors.mean()
            pct_5  = (errors < 5).mean() * 100
            pct_3  = (errors < 3).mean() * 100

            print(f"  {target:<25} {mae:>10.2f} {pct_5:>9.1f}% {pct_3:>9.1f}%")
            all_errors.extend(errors.tolist())

    if all_errors:
        print(f"\n  Overall MAE (all measurements): {np.mean(all_errors):.2f} cm")
        print(f"  Within 5 cm: {(np.array(all_errors) < 5).mean()*100:.1f}%")
        print(f"  Within 3 cm: {(np.array(all_errors) < 3).mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train BodyM measurement models')
    parser.add_argument('--data_dir',   default='./bodym',   help='Root of BodyM dataset')
    parser.add_argument('--output_dir', default='./models',  help='Where to save models')
    parser.add_argument('--max_rows',   type=int, default=None, help='Limit rows (for debugging)')
    parser.add_argument('--skip_images', action='store_true',   help='Skip MediaPipe (use cached features)')
    parser.add_argument('--cache_dir',  default='./cache',   help='Cache for extracted features')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Load labels ────────────────────────────────────────────────────
    train_csv = os.path.join(args.data_dir, 'train', 'train_labels.csv')
    test_a_csv = os.path.join(args.data_dir, 'test_a', 'test_a_labels.csv')
    test_b_csv = os.path.join(args.data_dir, 'test_b', 'test_b_labels.csv')

    df_train = load_labels(train_csv)

    # ── Build or load feature matrix ──────────────────────────────────
    train_cache = os.path.join(args.cache_dir, 'train_features.parquet')

    if os.path.exists(train_cache) and args.skip_images:
        print(f"\nLoading cached train features from {train_cache}")
        df_train_feats = pd.read_parquet(train_cache)
    else:
        print("\nExtracting features from training images...")
        images_dir = os.path.join(args.data_dir, 'train', 'images')
        df_train_feats = build_feature_matrix(df_train, images_dir, args.max_rows)
        df_train_feats.to_parquet(train_cache, index=False)
        print(f"Features cached → {train_cache}")

    # ── Train models ────────────────────────────────────────────────────
    print("\nTraining models...")
    results = train_gender_models(df_train_feats, args.output_dir)

    # Save training summary
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining summary saved → {summary_path}")

    # ── Evaluate on test sets ───────────────────────────────────────────
    for test_csv, set_name, subdir in [
        (test_a_csv, 'Test Set A (lab)',        'test_a'),
        (test_b_csv, 'Test Set B (in-the-wild)', 'test_b'),
    ]:
        if not os.path.exists(test_csv):
            print(f"\nSkipping {set_name} — {test_csv} not found")
            continue

        test_cache = os.path.join(args.cache_dir, f'{subdir}_features.parquet')
        if os.path.exists(test_cache) and args.skip_images:
            df_test_feats = pd.read_parquet(test_cache)
        else:
            df_test = load_labels(test_csv)
            images_dir = os.path.join(args.data_dir, subdir, 'images')
            df_test_feats = build_feature_matrix(df_test, images_dir, args.max_rows)
            df_test_feats.to_parquet(test_cache, index=False)

        evaluate_on_test_set(df_test_feats, args.output_dir, set_name)

    print("\n✓ Done. Models saved to:", args.output_dir)


if __name__ == '__main__':
    main()
