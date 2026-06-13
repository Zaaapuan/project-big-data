"""Open the Flask application inside a native pywebview window."""

from __future__ import annotations

from employee_app.api import create_app


def run_desktop() -> None:
    import webview

    app = create_app()
    webview.create_window(
        title="Dashboard Clustering Profil Karyawan",
        url=app,
        width=1360,
        height=900,
        min_size=(1024, 720),
        resizable=True,
        text_select=True,
    )
    webview.start(debug=False)
