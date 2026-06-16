# Capacity Test Plan

Purpose: estimate the maximum reliable lifting capacity of the current NICO right-hand grasp setup.

## Method

1. Measure each object weight using a scale.
2. Use the final calibrated grasp settings only.
3. Run 3 or 5 trials per object/weight level.
4. Keep object position and command phrase controlled.
5. Record first-person frames and manual labels for every trial.

## Success Definition

A capacity-test success means the object is enclosed and lifted/held for 2-3 seconds without slipping.

## Capacity Definition

Maximum reliable capacity is the highest tested weight with at least:

- 2/3 successful trials, or
- 3/5 successful trials.

## Current Data Status

Capacity data is `to be collected`. Existing logs do not include measured object weights, systematic weight levels, lift height, or hold time. Do not invent these values for AP2.

## CSV Template

Use `capacity_test_template.csv` to fill object name, weight, trial count, success/partial/failed counts, maximum lift height, hold time, and notes.
