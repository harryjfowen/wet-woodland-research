#!/usr/bin/env python3
"""
Preferred entrypoint for wet woodland MaxEnt potential modelling.

This thin wrapper preserves a short, stable script name while reusing the
existing implementation in ``run_elapid_potential.py``.
"""

from __future__ import annotations

from run_elapid_potential import main


if __name__ == "__main__":
    raise SystemExit(main())
