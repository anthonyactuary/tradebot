from __future__ import annotations

import os
import sys


def main() -> None:
    """Convenience launcher for the Streamlit dashboard.

    This is meant for VS Code's "Run Python File" button.
    """

    try:
        from streamlit.web.cli import main as streamlit_main
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "streamlit is not installed in the selected interpreter. "
            "Install it with: pip install streamlit"
        ) from e

    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")

    # Preserve any app args the user provides (after --), just like streamlit does.
    # Example:
    #   python analytics/run_app.py -- --fills_csv runs/kalshi_fills.csv
    user_args = []
    if "--" in sys.argv:
        user_args = sys.argv[sys.argv.index("--") + 1 :]

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--",
        *user_args,
    ]

    streamlit_main()


if __name__ == "__main__":
    main()
