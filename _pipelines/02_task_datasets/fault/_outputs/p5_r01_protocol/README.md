# Fault P5.1 R0/R1 portable evidence

This directory contains lane gates and a bounded development-only mechanism audit.
A/B intentionally use an invalid complement-as-negative label only to diagnose split leakage;
they are not rankable. C uses the legal mask and buffered split contract and stops before
model construction because audited negatives and two legal folds are absent. No holdout,
prediction archive, checkpoint, winner, HPO result, or official metric is present.
