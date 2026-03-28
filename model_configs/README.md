## Model Configs

Every `*.json` file in this folder is loaded automatically in alphabetical order.

You can keep multiple configs at once:
- one file for the base feature list
- one file for row filters
- one file for manual prediction overrides

Supported top-level keys:
- `enabled`: `true` or `false`
- `feature_columns`: replace the full kept feature list
- `add_feature_columns`: append features
- `remove_feature_columns`: remove features
- `categorical_feature_columns`: replace the full categorical list
- `add_categorical_feature_columns`: append categorical features
- `remove_categorical_feature_columns`: remove categorical features
- `row_filters`: rules that keep or drop rows before training/prediction splitting
- `prediction_overrides`: rules that override `NbPaxTotalPrediction` after model prediction

Prediction override actions:
- `set_prediction`: force a fixed value
- `set_prediction_from_column`: copy the prediction from another column on the same row
- `clip_upper_column`: cap the prediction so it does not exceed another column
- `clip_lower_column`: raise the prediction so it is not below another column

Overrides are applied in file order, then rule order.

`row_filters` support:
- `action: "drop"`: remove matching rows
- `action: "keep"`: keep matching rows

If at least one `keep` rule exists, only rows matching any keep rule are kept, then all `drop` rules are applied.

Condition format:

```json
{
  "all": [
    { "column": "NbOfSeats", "operator": "eq", "value": 0 },
    { "column": "Direction", "operator": "eq", "value": "D" }
  ]
}
```

Supported operators:
- `eq`, `==`
- `ne`, `!=`
- `gt`, `>`
- `gte`, `>=`
- `lt`, `<`
- `lte`, `<=`
- `in`
- `not_in`
- `is_null`
- `not_null`

Example row filter:

```json
{
  "name": "drop_zero_seats",
  "action": "drop",
  "when": {
    "column": "NbOfSeats",
    "operator": "eq",
    "value": 0
  }
}
```

Example prediction override:

```json
{
  "name": "force_zero_pax_when_zero_seats",
  "set_prediction": 0,
  "when": {
    "column": "NbOfSeats",
    "operator": "eq",
    "value": 0
  }
}
```

Example prediction cap:

```json
{
  "name": "cap_prediction_to_seats",
  "clip_upper_column": "NbOfSeats"
}
```
