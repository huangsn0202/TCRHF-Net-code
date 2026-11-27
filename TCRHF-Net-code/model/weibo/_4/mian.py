"""Compatibility entry point for the recurring 'mian.py' filename."""

try:
    from .main import main
except ImportError:
    from main import main

if __name__ == "__main__":
    main()
