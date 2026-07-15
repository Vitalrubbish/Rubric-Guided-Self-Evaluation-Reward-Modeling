# Method 1 Archive

Method 1 is no longer the active APPS route.

## Current Status

Method 1 explored direct generator/evaluator training with APPS repair and preference data. The strongest variants did not produce stable held-out gains:

- mixed-clean RS-SFT: `77 -> 76`
- strict v1.5: `76 -> 73`
- several DPO and same-problem variants improved one metric while damaging syntax, length, or pass rate

The route is therefore preserved as a negative-result archive rather than an active implementation path.

## Why Keep It

The archived Method 1 notes are still useful for:

- explaining why the project switched to Method 2;
- preventing repeated attempts at already-failed DPO/SFT variants;
- retaining lessons about syntax stability, length guards, and format extraction;
- supporting the final report's negative-results section.

## Active Route

Use `docs/method2/README.md` for the current APPS Method 2 route.

## Archive Contents

Detailed Method 1 experiment logs are under `docs/method1/archive/`.
