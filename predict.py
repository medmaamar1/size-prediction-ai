"""
Inference Module — predict body measurements from photos
=========================================================
Usage:
    from predict import MeasurementPredictor
    p = MeasurementPredictor('./models')
    result = p.predict(
        front_image_path='front.jpg',
        side_image_path='side.jpg',   # optional but recommended
        height_cm=175,
        weight_kg=75,
        gender='male',
        body_type='muscular',         # from dropdown
        muscle_level=2,               # 0=low, 1=mid, 2=high
        fat_level=0,
    )
    print(result)
"""

import os
import json
import math
import numpy as np
import cv2
import xgboost as xgb
import mediapipe as mp

from train import (
    extract_features_from_image,
    derive_body_type_scores,
    TARGETS,
)

mp_pose = mp.solutions.pose


# ── Body type dropdown → numeric encoding ───────────────────────────────────
BODY_TYPE_MAP_MALE = {
    'skinny':    0,
    'average':   1,
    'muscular':  2,
    'fat':       3,
    'skinny_fat': 4,
}
BODY_TYPE_MAP_FEMALE = {
    'slim':      0,
    'curvy':     1,
    'pear':      2,
    'apple':     3,
    'athletic':  4,
    'full_bust': 5,
}


def ellipse_circumference(width_cm, depth_cm):
    """Ramanujan ellipse approximation — accurate to <1mm for body shapes."""
    a = width_cm / 2
    b = depth_cm / 2
    if a <= 0 or b <= 0:
        return 0
    return math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))


class MeasurementPredictor:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self._models    = {}
        self._feat_cols = {}
        self._load_models()

    def _load_models(self):
        for gender in ['male', 'female']:
            feat_path = os.path.join(self.models_dir, f'{gender}_feature_cols.json')
            if not os.path.exists(feat_path):
                print(f"Warning: no model found for {gender} in {self.models_dir}")
                continue
            with open(feat_path) as f:
                self._feat_cols[gender] = json.load(f)

            self._models[gender] = {}
            for target in TARGETS:
                model_path = os.path.join(self.models_dir, f'{gender}_{target}.json')
                if os.path.exists(model_path):
                    m = xgb.XGBRegressor()
                    m.load_model(model_path)
                    self._models[gender][target] = m

        print(f"Loaded models for: {list(self._models.keys())}")

    def predict(
        self,
        front_image_path,
        height_cm,
        weight_kg,
        gender,                  # 'male' or 'female'
        side_image_path=None,
        body_type=None,          # from dropdown
        muscle_level=1,          # 0/1/2
        fat_level=1,             # 0/1/2
        bra_size=None,           # e.g. '36C' (women only)
    ):
        gender = gender.lower().strip()
        if gender not in self._models:
            raise ValueError(f"No model for gender '{gender}'. Available: {list(self._models.keys())}")

        bmi = weight_kg / (height_cm / 100) ** 2

        # ── Extract MediaPipe features ──────────────────────────────────
        front_feats = extract_features_from_image(front_image_path, height_cm, is_side=False)
        if front_feats is None:
            return {'error': 'MediaPipe failed on front photo — check pose visibility'}

        side_feats = {}
        has_side = False
        if side_image_path and os.path.exists(side_image_path):
            sf = extract_features_from_image(side_image_path, height_cm, is_side=True)
            if sf:
                side_feats = sf
                has_side = True

        # ── Static features ─────────────────────────────────────────────
        body_type_enc = 1  # default average
        if gender == 'male' and body_type:
            body_type_enc = BODY_TYPE_MAP_MALE.get(body_type.lower(), 1)
        elif gender == 'female' and body_type:
            body_type_enc = BODY_TYPE_MAP_FEMALE.get(body_type.lower(), 0)

        static = {
            'height_cm':      height_cm,
            'weight_kg':      weight_kg,
            'bmi':            bmi,
            'bmi_bucket':     min(3, int(bmi // 5) - 3),  # rough bucket
            'gender_enc':     1 if gender == 'female' else 0,
            'has_side':       int(has_side),
            'body_type_enc':  body_type_enc,
            'muscle_level':   muscle_level,
            'fat_level':      fat_level,
        }

        # ── Merge and derive scores ──────────────────────────────────────
        combined = {**static, **front_feats, **side_feats}
        scores   = derive_body_type_scores(combined)
        combined.update(scores)

        # ── Run XGBoost predictions ──────────────────────────────────────
        feat_cols = self._feat_cols[gender]
        X = np.array([[combined.get(c, 0) for c in feat_cols]])

        predictions = {}
        for target, model in self._models[gender].items():
            predictions[target] = round(float(model.predict(X)[0]), 1)

        # ── Circumferences from ellipse (front + side) ───────────────────
        # When side photo available: much better circumference estimate
        if has_side:
            ppc_f = front_feats.get('front_px_per_cm', 1)
            ppc_s = side_feats.get('side_px_per_cm', 1)

            def to_cm_side(key):
                return side_feats.get(f'side_mask_{key}_cm', 0)

            def to_cm_front(key):
                return front_feats.get(f'front_mask_{key}_cm', 0)

            chest_circ = ellipse_circumference(to_cm_front('chest'), to_cm_side('chest'))
            waist_circ = ellipse_circumference(to_cm_front('waist'), to_cm_side('waist'))
            hip_circ   = ellipse_circumference(to_cm_front('hip'),   to_cm_side('hip'))

            # Blend XGBoost prediction (30%) + ellipse (70%) when side available
            if chest_circ > 0:
                predictions['chest_cm'] = round(0.3 * predictions.get('chest_cm', chest_circ) + 0.7 * chest_circ, 1)
            if waist_circ > 0:
                predictions['waist_cm'] = round(0.3 * predictions.get('waist_cm', waist_circ) + 0.7 * waist_circ, 1)
            if hip_circ > 0:
                predictions['hip_cm']   = round(0.3 * predictions.get('hip_cm',   hip_circ)   + 0.7 * hip_circ,   1)

        # ── Bust detection (women) ───────────────────────────────────────
        bust_score = None
        if gender == 'female':
            bust_score = self._detect_bust(
                front_feats, side_feats, bra_size, body_type
            )
            predictions['bust_score'] = bust_score

        # ── Body shape scores ────────────────────────────────────────────
        predictions['fat_score']      = scores['fat_score']
        predictions['muscle_score']   = scores['muscle_score']
        predictions['skinny_fat_score'] = scores['skinny_fat_score']

        # ── Fit warnings ─────────────────────────────────────────────────
        predictions['warnings'] = self._generate_warnings(predictions, gender)

        # ── Confidence ───────────────────────────────────────────────────
        predictions['accuracy_mode'] = 'front+side (high)' if has_side else 'front only (medium)'

        return predictions

    def _detect_bust(self, front_feats, side_feats, bra_size, body_type):
        """Bust score 1–5. Priority: bra size → side photo → front mask → body type."""
        # Priority 1: user entered bra size
        if bra_size:
            cup = bra_size.strip()[-1].upper()
            cup_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 5, 'G': 5}
            return cup_map.get(cup, 2)

        # Priority 2: side photo protrusion
        if side_feats:
            chest_d = side_feats.get('side_mask_chest_cm', 0)
            waist_d = side_feats.get('side_mask_waist_cm', 0)
            if chest_d > 0 and waist_d > 0:
                protrusion = (chest_d - waist_d) / waist_d
                if protrusion > 0.20:  return 5
                if protrusion > 0.14:  return 4
                if protrusion > 0.08:  return 3
                return 2

        # Priority 3: front mask ratio
        chest_w = front_feats.get('front_mask_chest_cm', 0)
        waist_w = front_feats.get('front_mask_waist_cm', 1)
        if chest_w > 0 and waist_w > 0:
            ratio = chest_w / waist_w
            if ratio > 1.30:  return 4
            if ratio > 1.18:  return 3
            if ratio > 1.08:  return 2
            return 1

        # Priority 4: body type checkbox
        if body_type and 'full_bust' in str(body_type).lower():
            return 4

        return 2  # default medium

    def _generate_warnings(self, preds, gender):
        """Human-readable fit warnings based on body shape scores."""
        warnings = []

        if preds.get('muscle_score', 0) >= 5:
            warnings.append(
                "Muscular build — standard size fits waist but may be tight at "
                "shoulders and chest. Consider sizing up on top."
            )

        if preds.get('skinny_fat_score', 0) >= 3:
            warnings.append(
                "Soft midsection detected — sized up at waist while keeping "
                "chest/shoulder at your measured size."
            )

        if gender == 'female':
            bust = preds.get('bust_score', 2)
            if bust >= 4:
                warnings.append(
                    "Larger bust — check brand's bust measurement before ordering. "
                    "May need to size up on top."
                )
            body_type_body = preds.get('body_type', '')
            waist = preds.get('waist_cm', 0)
            hip   = preds.get('hip_cm', 0)
            if hip > 0 and waist > 0 and (hip - waist) > 25:
                warnings.append(
                    "Significant hip-to-waist difference — consider sizing for hips "
                    "and having the waist taken in, or look for stretch fabrics."
                )

        if preds.get('fat_score', 0) >= 5:
            warnings.append(
                "Sized for your waist measurement — double-check hip measurement "
                "if ordering trousers or skirts."
            )

        return warnings
