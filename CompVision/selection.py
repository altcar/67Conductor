import threading
import json
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import math
import time

#!/usr/bin/env python3
"""
selection.py

Tkinter GUI with a modern/futuristic look:
 - Items rendered as centered "flex" cards in a scrollable canvas
 - Hover a card to show a floating detail tooltip that follows the mouse
 - Click to select (toggle) items; multiple selection supported
 - GET / POST to configured endpoints

Modified: single-column "body" layout, no right column; everything centered.

Behavior tweak: do not show error message boxes. messagebox.showerror is monkey-patched
to write the error into the app status bar (if available) or print as a fallback.

Additional: attach an OpenCV (cv2) video stream rendered into the right half
of the main window. This attaches automatically after the Tk root is created.
"""
# Optional imports for video; fail gracefully if missing.
try:
    import cv2
    from PIL import Image, ImageTk
    import time
except Exception:
    cv2 = None
    Image = None
    ImageTk = None

# Adjust header/buttons/title alignment to left after the main Tk root and widgets exist.
def _left_align_controls_when_ready():
    def waiter():
        root = None
        for _ in range(200):  # ~10s timeout
            root = getattr(tk, "_default_root", None)
            if root:
                break
            time.sleep(0.05)
        if not root:
            return

        def find_by_text(widget, text):
            try:
                txt = None
                try:
                    txt = widget.cget("text")
                except Exception:
                    pass
                if txt == text:
                    try:
                        # If this is the "Send Selected" button, enhance its behavior:
                        # - clear categories/UI immediately
                        # - run the normal send flow
                        # - after send completes (button re-enabled) trigger a fetch to repopulate items
                        root_app = getattr(tk, "_default_root", None)
                        if text == "Send Selected" and root_app is not None and not getattr(widget, "_send_wrapped", False):
                            def _on_send_click():
                                try:
                                    # clear items/UI on the main thread
                                    def _clear_ui():
                                        try:
                                            root_app.items = []
                                            root_app.selected_indices.clear()
                                            try:
                                                root_app.populate_items()
                                            except Exception:
                                                pass
                                            try:
                                                root_app.set_status("Cleared categories, sending selection...")
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                    try:
                                        root_app.after(0, _clear_ui)
                                    except Exception:
                                        _clear_ui()

                                    # call original send handler (SelectionApp method)
                                    try:
                                        root_app.send_selected_async()
                                    except Exception:
                                        pass

                                    # wait for the send button to become enabled again, then fetch items
                                    def _wait_then_fetch():
                                        btn = widget
                                        # wait up to ~10s
                                        for _ in range(200):
                                            try:
                                                # ttk.Button supports instate
                                                if hasattr(btn, "instate"):
                                                    if not btn.instate(["disabled"]):
                                                        break
                                                else:
                                                    if str(btn.cget("state")) != "disabled":
                                                        break
                                            except Exception:
                                                pass
                                            time.sleep(0.05)
                                        try:
                                            root_app.fetch_items_async()
                                        except Exception:
                                            pass
                                    threading.Thread(target=_wait_then_fetch, daemon=True).start()
                                except Exception:
                                    pass

                            try:
                                # Prefer setting command (ttk.Button). Fallback to binding click.
                                widget.config(command=_on_send_click)
                            except Exception:
                                try:
                                    widget.bind("<Button-1>", lambda e: _on_send_click(), add="+")
                                except Exception:
                                    pass
                            try:
                                widget._send_wrapped = True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return widget
            except Exception:
                pass
            for child in widget.winfo_children():
                found = find_by_text(child, text)
                if found:
                    return found
            return None

        def adjust():
            try:
                refresh = find_by_text(root, "Refresh")
                send = find_by_text(root, "Send Selected")
                items_lbl = find_by_text(root, "Items")

                # If the buttons exist, ensure their parent frame is left-anchored
                for btn in (refresh, send):
                    if btn is not None:
                        parent = getattr(btn, "master", None)
                        if parent is not None:
                            try:
                                parent.pack_configure(anchor="w")
                            except Exception:
                                pass
                        try:
                            btn.pack_configure(side=tk.LEFT)
                        except Exception:
                            pass

                # Align the "Items" label to the left
                if items_lbl is not None:
                    try:
                        items_lbl.pack_configure(anchor="w")
                    except Exception:
                        pass

            except Exception:
                pass

        # Schedule an adjustment shortly on the main thread (allow create_widgets to run)
        try:
            root.after(150, adjust)
        except Exception:
            adjust()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

_left_align_controls_when_ready()


# Video streamer that updates a Tk label with frames from cv2.VideoCapture
class _VideoStreamer:
    def __init__(self, root, label, src=0, fps=20):
        self.root = root
        self.label = label
        self.src = src
        self.fps = max(1, fps)
        self._running = False
        self._cap = None
        self._thread = None

    def start(self):
        if cv2 is None or Image is None or ImageTk is None:
            return
        if self._running:
            return
        try:
            self._cap = cv2.VideoCapture(self.src)
            # small probe: set a reasonable frame size if camera supports it
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            except Exception:
                pass
            if not self._cap.isOpened():
                # early abort; ensure resources freed
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                return
        except Exception:
            self._cap = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # allow thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        # clear label image
        try:
            self.root.after(0, lambda: (self.label.config(image=""), setattr(self.label, "image", None)))
        except Exception:
            pass

    def _run_loop(self):
        interval = 1.0 / self.fps
        while self._running and self._cap:
            t0 = time.time()
            ret, frame = False, None
            try:
                ret, frame = self._cap.read()
            except Exception:
                ret = False
            if not ret or frame is None:
                # no frame: wait a bit and continue
                time.sleep(interval)
                continue
            try:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                # scale to fit half-window width while preserving aspect
                # compute target width from root current width (may be zero early)
                w = max(160, int(self.root.winfo_width() * 0.5))
                h = max(120, int(self.root.winfo_height()))
                img.thumbnail((w - 16, h - 16), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                # update label on main thread
                def _update():
                    try:
                        self.label.config(image=imgtk)
                        self.label.image = imgtk  # keep reference
                    except Exception:
                        pass
                self.root.after(0, _update)
            except Exception:
                pass
            dt = time.time() - t0
            sleep_for = max(0, interval - dt)
            time.sleep(sleep_for)

# Poll for the Tk root being created and then attach a video panel to the right half.
def _attach_video_when_ready(src=0, fps=20):
    if cv2 is None or Image is None or ImageTk is None:
        # cv2/PIL not available; silently skip
        return

    def waiter():
        # wait until the main Tk root exists
        root = None
        for _ in range(200):  # ~20s timeout
            root = getattr(tk, "_default_root", None)
            if root:
                break
            time.sleep(0.1)
        if not root:
            return

        def attach():
            try:
                # create an anchored right-half frame using place so it doesn't require access to other local frames
                right = tk.Frame(root, bg=NEON_BG, highlightthickness=0)
                right.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=1.0)

                # optional header in video panel
                hdr = tk.Frame(right, bg=NEON_BG)
                hdr.pack(fill=tk.X, pady=(8, 4))
                ttk.Label(hdr, text="Live Video", background=NEON_BG, foreground=ACCENT, font=("Segoe UI", 11, "bold")).pack()

                # video display label
                video_lbl = tk.Label(right, bg=CARD_BG)
                video_lbl.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

                streamer = _VideoStreamer(root, video_lbl, src=src, fps=fps)
                streamer.start()

                # ensure streamer stops when root is destroyed
                def _on_destroy(event):
                    if event.widget is root:
                        try:
                            streamer.stop()
                        except Exception:
                            pass
                root.bind("<Destroy>", _on_destroy, add="+")
            except Exception:
                # do not crash the app if video attachment fails
                pass

        # schedule attach on main thread
        try:
            root.after(100, attach)
        except Exception:
            attach()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

# Start the auto-attach in background (non-blocking). Change src index if you want another camera or a video file path.
_attach_video_when_ready(src=0, fps=20)
# Route messagebox.showerror to the app status bar (silent, non-modal)
_original_showerror = messagebox.showerror

def _showerror_to_status(title, message, **kwargs):
    try:
        root = getattr(tk, "_default_root", None)
        if root and hasattr(root, "set_status"):
            # ensure status update runs on the main thread
            root.after(0, lambda: root.set_status(f"{title}: {message}"))
        else:
            # fallback: print to stdout (non-blocking)
            print(f"{title}: {message}")
    except Exception:
        # swallow any issues to keep behavior silent
        pass

messagebox.showerror = _showerror_to_status
# ...existing code...
categories = [
  { "name": "Jeans", "picurl": "lib/pic/jean" },
  { "name": "BB", "picurl": "lib/pic/jean" },
  { "name": "T-Shirt", "picurl": "lib/pic/jean" },
  { "name": "Hoodie", "picurl": "lib/pic/jean" },
  { "name": "Sneakers", "picurl": "lib/pic/jean" },
  { "name": "Jacket", "picurl": "lib/pic/jean" },
  { "name": "Coat", "picurl": "lib/pic/jean" },
  { "name": "Dress", "picurl": "lib/pic/jean" },
  { "name": "Skirt", "picurl": "lib/pic/jean" },
  { "name": "Shorts", "picurl": "lib/pic/jean" },
  { "name": "Blouse", "picurl": "lib/pic/jean" },
  { "name": "Shirt", "picurl": "lib/pic/jean" },
  { "name": "Scarf", "picurl": "lib/pic/jean" },
  { "name": "Hat", "picurl": "lib/pic/jean" },
  { "name": "Belt", "picurl": "lib/pic/jean" },
  { "name": "Socks", "picurl": "lib/pic/jean" },
  { "name": "Boots", "picurl": "lib/pic/jean" },
  { "name": "Gloves", "picurl": "lib/pic/jean" },
  { "name": "Sunglasses", "picurl": "lib/pic/jean" },
  { "name": "Watch", "picurl": "lib/pic/jean" }
]
# ...existing code...

# Configure endpoints
GET_URL = "http://localhost:8080/api/items"
POST_URL = "http://localhost:8080/api/submit"
REQUEST_TIMEOUT = 10

NEON_BG = "#0b0f1a"
CARD_BG = "#0f1724"
CARD_HOVER = "#0b1220"
ACCENT = "#39ffca"
TEXT = "#e6eef8"
TOOLTIP_BG = "#071019"

class SelectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("API Selection GUI — Futuristic")
        self.geometry("900x600")
        self.config(bg=NEON_BG)
        self.items = []
        self.selected_indices = set()
        self.item_widgets = {}
        self.tooltip = None

        self._setup_style()
        self.create_widgets()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=NEON_BG)
        style.configure("Header.TButton", background=NEON_BG, foreground=ACCENT)
        style.configure("Status.TLabel", background=NEON_BG, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.map("Header.TButton", background=[("active", "#071019")])

        # Ensure we initialize the UI with the default category list after widgets are created.
        # _setup_style() is called before create_widgets() in __init__, so schedule the init
        # for idle time to guarantee widgets exist.
        self.after(0, self._init_with_category)

    def _init_with_category(self):
        # Initialize items from the module-level `categories` list and populate the UI.
        try:
            # shallow copy to avoid accidental shared-mutation
            self.items = list(categories)
        except Exception:
            self.items = []
        # Replace populate_items with a layout that will wrap into centered columns

        def populate_items_multi():
            # Ensure widget sizes are available; if canvas height is not ready, retry shortly.
            ch = self.canvas.winfo_height()
            if ch <= 1:
                self.after(50, populate_items_multi)
                return

            # Clear previous
            for w in self.items_container.winfo_children():
                w.destroy()
            self.item_widgets.clear()
            self.selected_indices.clear()

            # reserve bottom margin = 5% of canvas height
            bottom_margin = max(20, int(ch * 0.05))
            # approximate card height + vertical padding (tune as needed)
            card_h = 56
            vpad = 12
            available_h = max(1, ch - bottom_margin - 40)  # extra top padding
            rows_per_col = max(1, available_h // (card_h + vpad))
            cols = max(1, math.ceil(len(self.items) / rows_per_col))

            # central wrapper so columns stay centered
            center = tk.Frame(self.items_container, bg=NEON_BG)
            center.pack(anchor="n", pady=(8, bottom_margin))

            # build grid: rows_per_col rows x cols columns
            for idx, item in enumerate(self.items):
                col = idx // rows_per_col
                row = idx % rows_per_col

                # create card in grid cell
                cell = tk.Frame(center, bg=NEON_BG)
                cell.grid(row=row, column=col, padx=10, pady=6)

                card = tk.Frame(cell, bg=CARD_BG, bd=1, relief=tk.FLAT, highlightthickness=0)
                card.pack(anchor="center", ipadx=12, ipady=10)
                label_text = self.item_label(item, idx)
                lbl = tk.Label(card, text=label_text, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 11, "bold"))
                lbl.pack()

                # store widget refs
                self.item_widgets[idx] = (card, lbl)

                # bindings for click and hover
                for widget in (card, lbl, cell):
                    widget.bind("<Button-1>", lambda e, ix=idx: self.toggle_select(ix))
                    widget.bind("<Enter>", lambda e, ix=idx: self.on_item_enter(ix, e))
                    widget.bind("<Leave>", lambda e, ix=idx: self.on_item_leave(ix, e))
                    widget.bind("<Motion>", lambda e, ix=idx: self.on_item_motion(ix, e))

            # make columns compact and keep grid centered
            for c in range(cols):
                center.grid_columnconfigure(c, weight=1)

            # update scrollregion after layout
            self.after(50, lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # assign and run
        self.populate_items = populate_items_multi
        self.populate_items()
        self.set_status(f"Initialized with {len(self.items)} categories")

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Top: centered control buttons
        top = ttk.Frame(frm)
        top.pack(fill=tk.X, pady=(0,8))
        header_btn_frame = ttk.Frame(top)
        header_btn_frame.pack()  # centered by default
        self.refresh_btn = ttk.Button(header_btn_frame, text="Refresh", command=self.fetch_items_async, style="Header.TButton")
        self.refresh_btn.pack(side=tk.LEFT, padx=8)
        self.send_btn = ttk.Button(header_btn_frame, text="Send Selected", command=self.send_selected_async, style="Header.TButton")
        self.send_btn.pack(side=tk.LEFT, padx=8)

        # Body: single centered column (no right column)
        body = ttk.Frame(frm)
        body.pack(fill=tk.BOTH, expand=True)

        lbl = ttk.Label(body, text="Items", background=NEON_BG, foreground=ACCENT, font=("Segoe UI", 12, "bold"))
        lbl.pack(anchor="center")

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        self.canvas = tk.Canvas(canvas_frame, bg=NEON_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.items_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.items_container, anchor="n")
        self.items_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Status bar
        self.status = ttk.Label(self, text="Ready", style="Status.TLabel", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def set_status(self, msg):
        self.status.config(text=msg)

    def fetch_items_async(self):
        def worker():
            self.after(0, lambda: self.set_status("Fetching items..."))
            self.after(0, lambda: self.refresh_btn.config(state=tk.DISABLED))
            try:
                r = requests.get(GET_URL, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                items = self.normalize_items(data)
                self.items = items
                self.after(0, self.populate_items)
                self.after(0, lambda: self.set_status(f"Fetched {len(items)} items"))
            except Exception as e:
                self.after(0, lambda: self.set_status(f"Failed to fetch: {e}"))
                self.after(0, lambda: messagebox.showerror("Fetch error", str(e)))
            finally:
                self.after(0, lambda: self.refresh_btn.config(state=tk.NORMAL))
        threading.Thread(target=worker, daemon=True).start()

    def normalize_items(self, data):
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "data", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return [data]

    def populate_items(self):
        # clear previous
        for w in self.items_container.winfo_children():
            w.destroy()
        self.item_widgets.clear()
        self.selected_indices.clear()

        # create centered "cards" using a container and pack center
        for i, item in enumerate(self.items):
            wrapper = tk.Frame(self.items_container, bg=NEON_BG)
            wrapper.pack(fill=tk.X, pady=6, padx=6)

            card = tk.Frame(wrapper, bg=CARD_BG, bd=1, relief=tk.FLAT, highlightthickness=0)
            card.pack(anchor="center", ipadx=12, ipady=10, fill=tk.X, padx=200)  # padded to keep center feel
            label_text = self.item_label(item, i)
            lbl = tk.Label(card, text=label_text, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 11, "bold"))
            lbl.pack()

            # store widget refs
            self.item_widgets[i] = (card, lbl)

            # bindings for click and hover
            for widget in (card, lbl, wrapper):
                widget.bind("<Button-1>", lambda e, idx=i: self.toggle_select(idx))
                widget.bind("<Enter>", lambda e, idx=i: self.on_item_enter(idx, e))
                widget.bind("<Leave>", lambda e, idx=i: self.on_item_leave(idx, e))
                widget.bind("<Motion>", lambda e, idx=i: self.on_item_motion(idx, e))

        # small delay to set scrollregion
        self.after(50, lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def item_label(self, item, idx):
        if isinstance(item, dict):
            for k in ("id", "name", "title"):
                if k in item:
                    return f"{item.get(k)}"
            # show a compact preview
            preview = ", ".join(f"{k}={v}" for k, v in list(item.items())[:3])
            return f"{preview}"
        return repr(item)

    def toggle_select(self, idx):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.add(idx)
        self.update_item_visual(idx)
        # reflect selection count in status
        self.set_status(f"Selected {len(self.selected_indices)} item(s)")

    def update_item_visual(self, idx):
        card, lbl = self.item_widgets.get(idx, (None, None))
        if not card:
            return
        if idx in self.selected_indices:
            card.config(bg=ACCENT)
            lbl.config(bg=ACCENT, fg="#001518")
        else:
            card.config(bg=CARD_BG)
            lbl.config(bg=CARD_BG, fg=TEXT)

    def on_item_enter(self, idx, event):
        item = self.items[idx]
        text = json.dumps(item, indent=2, ensure_ascii=False)
        self.show_tooltip(text, event)

    def on_item_motion(self, idx, event):
        # move tooltip with pointer
        if self.tooltip:
            x = self.winfo_pointerx() + 18
            y = self.winfo_pointery() + 18
            self.tooltip.geometry(f"+{x}+{y}")

    def on_item_leave(self, idx, event):
        self.hide_tooltip()

    def show_tooltip(self, text, event):
        self.hide_tooltip()
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.attributes("-topmost", True)
        self.tooltip.config(bg=TOOLTIP_BG)
        # content
        label = tk.Label(self.tooltip, text=text, justify=tk.LEFT, bg=TOOLTIP_BG, fg=ACCENT,
                         font=("Consolas", 10), bd=1, padx=8, pady=6)
        label.pack()
        # position near pointer
        x = self.winfo_pointerx() + 18
        y = self.winfo_pointery() + 18
        self.tooltip.geometry(f"+{x}+{y}")

    def hide_tooltip(self):
        if self.tooltip:
            try:
                self.tooltip.destroy()
            except Exception:
                pass
            self.tooltip = None

    def send_selected_async(self):
        if not self.selected_indices:
            messagebox.showinfo("No selection", "Please select at least one item to send.")
            return
        selected_items = [self.items[i] for i in sorted(self.selected_indices)]

        def worker():
            self.after(0, lambda: self.set_status("Sending selected items..."))
            self.after(0, lambda: self.send_btn.config(state=tk.DISABLED))
            try:
                payload = {"selected": selected_items}
                r = requests.post(POST_URL, json=payload, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                try:
                    resp = r.json()
                    msg = json.dumps(resp, indent=2, ensure_ascii=False)
                except Exception:
                    msg = r.text
                self.after(0, lambda: messagebox.showinfo("Success", f"Server response:\n{msg}"))
                self.after(0, lambda: self.set_status("Send succeeded"))
            except Exception as e:
                self.after(0, lambda: self.set_status(f"Send failed: {e}"))
                self.after(0, lambda: messagebox.showerror("Send error", str(e)))
            finally:
                self.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    app = SelectionApp()
    app.mainloop()
