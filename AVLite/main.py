from c50_visualization.c51_visualizer_app import VisualizerApp
import logging

log = logging.getLogger(__name__)


if __name__ == "__main__":
    source_run = True

    import platform

    if platform.system() == "Linux":
        import os
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"
    else:
        import ctypes

        try:  # >= win 8.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:  # win 8.0 or less
            ctypes.windll.user32.SetProcessDPIAware()
        import os
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"

    app = VisualizerApp()
    app.mainloop()
