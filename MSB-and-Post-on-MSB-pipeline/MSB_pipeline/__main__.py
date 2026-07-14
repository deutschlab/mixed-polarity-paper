try:
    from .pipeline import main
except ImportError:  # pragma: no cover - allows running as a script
    from pipeline import main


if __name__ == "__main__":
    main()
