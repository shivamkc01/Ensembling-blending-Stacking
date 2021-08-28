# Ensembling-blending-Stacking
<img src="Ensemble_blending_Stacking.png" />

Stacking is not rocket science. It’s straightforward. If you have correct cross- validation and keep the folds same throughout the journey of your modelling task, nothing should overfit.
Let me describe the idea to you in simple points.
- Divide the training data into folds.
- Train a bunch of models: M1, M2.....Mn.
- Create full training predictions (using out of fold training) and test
predictions using all these models.
- Till here it is Level – 1 (L1).
- Use the fold predictions from these models as features to another model.
This is now a Level – 2 (L2) model.
- Use the same folds as before to train this L2 model.
- Now create OOF (out of fold) predictions on the training set and the test
set.
- Now you have L2 predictions for training data and also the final test set
predictions.
