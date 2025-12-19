# YOLO Homework Fixes Summary

## Issues Identified
1. **Bounding boxes barely visible after NMS** - Model predictions have very low confidence early in training
2. **PR curves showing no data** - Empty or near-empty prediction sets

## Fixes Applied

### 1. Forward Pass Constraints (Cell 14 - YOLO model)
✅ **Already applied**: Added `out[:, 1:5] = torch.sigmoid(out[:, 1:5])` to constrain x,y,w,h to [0,1]

### 2. Loss Function Stabilization (Cell 12)
✅ **Already applied**:
- Objectness uses sigmoid: `C_pred = torch.sigmoid(C_pred_raw)`
- Class predictions use softmax: `p_cls_prob = torch.softmax(p_cls, dim=-1)`

### 3. Learning Rate
✅ **Already lowered**: Changed from 1e-2 to 5e-3 in Cell 26

### 4. Part 7 Visualization - Need to apply these changes:

**Change NMS thresholds to be more permissive:**
- In visualization code, change `iou_thresh=0.5` to `iou_thresh=0.2` or `0.3`
- Change score threshold from `0.01` to `0.001` or `0.005`

**Use full validation set for PR curves:**
- Instead of sampled subset, iterate through entire `val_ds`
- Print diagnostic counts: GT boxes and predicted boxes per class

## Recommended Actions

1. **Retrain the model** from scratch with current settings (restart kernel, run all cells)
2. **Train longer** - Consider 40-60 epochs instead of 20 if mAP is still very low
3. **Check one visualization** - Run Part 6 (single image viz) to see if boxes make sense
4. **Part 7 fixes** - Update the thresholds as noted above

## Expected Behavior After Fixes

- Loss should converge to ~10-20 range
- mAP@0.3 should show gradual improvement over epochs (may start near zero, climb to 0.01-0.10+)
- Visualizations should show some boxes (even if not perfect) after 20 epochs
- PR curves should populate with at least some points for classes present in validation set













