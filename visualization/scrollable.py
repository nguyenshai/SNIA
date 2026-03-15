import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

def show_scrollable(fig, ani=None):
    """Wraps a matplotlib figure in a scrollable Tkinter window."""
    root = tk.Tk()
    root.title(fig.canvas.manager.get_window_title() or "Optimization Visualization")
    
    # Focus on maximizing the window / going fullscreen for better real-time overview
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w = min(1200, sw - 100)
    h = min(900, sh - 100)
    root.geometry(f"{w}x{h}")
    
    # Try to maximize the window natively on OS
    try:
        root.state('zoomed')
    except tk.TclError:
        root.attributes('-zoomed', True)


    # Set backgrounds
    bg_color = fig.patch.get_facecolor()
    # convert matplotlib color to hex
    try:
        import matplotlib.colors as mcolors
        bg_hex = mcolors.to_hex(bg_color)
    except:
        bg_hex = "#0d1117"

    root.configure(bg=bg_hex)

    canvas_container = tk.Canvas(root, bg=bg_hex, highlightthickness=0)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas_container.yview)
    hscrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas_container.xview)
    scrollable_frame = tk.Frame(canvas_container, bg=bg_hex)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_container.configure(
            scrollregion=canvas_container.bbox("all")
        )
    )

    canvas_container.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_container.configure(yscrollcommand=scrollbar.set, xscrollcommand=hscrollbar.set)

    canvas_container.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    hscrollbar.grid(row=1, column=0, sticky="ew")

    def _on_mousewheel(event):
        try:
            if event.delta:
                canvas_container.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

    def _on_mousewheel_h(event):
        try:
            if event.delta:
                canvas_container.xview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

    # Bind mouse wheel
    root.bind_all("<MouseWheel>", _on_mousewheel)
    root.bind_all("<Shift-MouseWheel>", _on_mousewheel_h)
    def _on_linux_scroll_up(event):
        canvas_container.yview_scroll(-1, "units")
    def _on_linux_scroll_down(event):
        canvas_container.yview_scroll(1, "units")
    root.bind_all("<Button-4>", _on_linux_scroll_up)
    root.bind_all("<Button-5>", _on_linux_scroll_down)

    # Embed figure
    canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_fig.draw()
    canvas_widget = canvas_fig.get_tk_widget()
    
    dpi = fig.get_dpi()
    fig_w, fig_h = fig.get_size_inches()
    canvas_widget.configure(width=int(fig_w * dpi), height=int(fig_h * dpi))
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Wrap the toolbar inside the main window, out of the scrollable frame, 
    # so it stays visible!
    toolbar_frame = tk.Frame(root, bg=bg_hex)
    toolbar_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
    toolbar = NavigationToolbar2Tk(canvas_fig, toolbar_frame)
    toolbar.update()
    
    root._ani_ref = ani
    
    def on_closing():
        root.quit()
        root.destroy()
        plt.close(fig)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
