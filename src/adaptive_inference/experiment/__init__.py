"""Phase 6 experiment layer: hold-out runs of the required systems.

Consumes the Phase 5 frozen calibration artifact (read-only) and wires
the existing single-pass and adaptive runners — plus the random-escalation
variant — to execute every required system on the held-out eval split
and emit a run manifest.
"""
