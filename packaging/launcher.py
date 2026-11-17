"""Point d'entrée des exécutables : démarre l'application et ouvre le navigateur.

Fermez la fenêtre du terminal pour arrêter l'application.
"""

import socket
import sys
import threading
import webbrowser
from pathlib import Path


def free_port(preferred: int = 8501) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Aucun port réseau disponible")


def main() -> None:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    port = free_port()
    url = f"http://localhost:{port}"

    print("=" * 60)
    print(" Quantique vs classique")
    print(f" L'application s'ouvre dans votre navigateur : {url}")
    print(" Fermez cette fenêtre pour arrêter l'application.")
    print("=" * 60)

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(base / "app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
        "--theme.base=light",
        "--theme.primaryColor=#6a4c93",
    ]
    threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
