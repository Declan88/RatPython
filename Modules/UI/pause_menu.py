"""
The in-game pause menu (ESC): resume, disconnect from the lobby (back to the main menu) or quit.
The game keeps running behind it - it's a multiplayer game, nothing can be frozen.
"""

from . import theme
from .widgets import Anchor, Button, Label, Panel


class PauseMenu(Panel):
    def __init__(self, on_resume, on_disconnect, on_quit, **kw):
        kw.setdefault("size_frac", (1.0, 1.0))
        super().__init__(color=(0, 0, 0, 150), visible=False, **kw)
        card = self.add(Panel(color=theme.CARD, anchor=Anchor.CENTER, layout="vertical", spacing=14,
                              padding=28, align="center", fit_content=True))
        card.add(Label("PAUSED", font_size=64, color=theme.ACCENT, font=theme.title_font_path()))
        card.add(Panel(size=(1, 8)))
        for text, callback in (("RESUME", on_resume), ("DISCONNECT", on_disconnect), ("QUIT GAME", on_quit)):
            card.add(Button(text, on_click=callback, size=(420, 72), font_size=34, **theme.BUTTON))
