# sturdy-adventure

code sample for sequential training.

`train_sequentially.lgb_train_sequentially` trains lightgbm models additively according to `feature_names_sequence`.

Suppose `feature_names_sequence = [cols_subset_0, cols_subset_1, ...]`, then training proceeds like so:

```
cols_cumulative = None
model = None

for cols_subset in feature_names_sequence:

    cols_cumulative += cols_subset
    model <- ModelClass.train(initial_model = model, cols_cumulative)
```