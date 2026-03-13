"""
TSP Visualization — Fullscreen stretched map + playback controls.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider

# ── Theme ────────────────────────────────────────────────────────────
BG    = '#0d1117'
PANEL = '#161b22'
GRID  = '#30363d'
TXT   = '#c9d1d9'
BLUE  = '#58a6ff'
CYAN  = '#39c5cf'
GREEN = '#3fb950'
RED   = '#ff7b72'
WHITE = '#ffffff'

EDGE_HIGHWAY  = '#00d4aa'
EDGE_NORMAL   = '#c9d1d9'
EDGE_MOUNTAIN = '#ff6b6b'

def _tc(tmult):
    if tmult < 0.9: return EDGE_HIGHWAY
    if tmult > 1.5: return EDGE_MOUNTAIN
    return EDGE_NORMAL

def _tl(tmult):
    if tmult < 0.9: return 'hwy'
    if tmult > 1.5: return 'mtn'
    return ''


def visualize_tsp_optimization(problem, AlgClass, algo_name='ACO',
                               params=None, iterations=80, interval=200,
                               save_path=None, show=True, max_frames=None):
    algo = AlgClass(problem, params=params or {})
    algo.solve(iterations=iterations)
    history = algo.history
    if not history:
        raise ValueError('No history')

    n_history = len(history)
    if max_frames and n_history > max_frames:
        idx = np.linspace(0, n_history - 1, max_frames, dtype=int).tolist()
        history = [history[i] for i in idx]
        n_history = len(history)

    cities = problem.cities
    n = problem.n_cities
    cx = np.array([c.x for c in cities])
    cy = np.array([c.y for c in cities])
    pad_x = max((cx.max() - cx.min()) * 0.15, 20)
    pad_y = max((cy.max() - cy.min()) * 0.15, 20)
    xlo, xhi = cx.min() - pad_x, cx.max() + pad_x
    ylo, yhi = cy.min() - pad_y, cy.max() + pad_y
    span = max(xhi - xlo, yhi - ylo)

    # ── Sizing ───────────────────────────────────────────────────
    NODE_S   = max(80, min(450, 4000 / n))
    DEPOT_S  = NODE_S * 1.3
    ID_FONT  = max(8, min(13, int(17 - n * 0.4)))
    EDGE_LW  = max(1.5, min(4.0, 40 / n))
    COST_FONT = max(7, min(12, int(15 - n * 0.3)))

    # ── Figure — map (left 80%) + info panel (right 20%) ────────
    fig = plt.figure(figsize=(28, 14))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1],
                          left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor('#161b22')
    ax_info.axis('off')

    # Maximize/Fullscreen window
    try:
        mgr = plt.get_current_fig_manager()
        try:
            mgr.full_screen_toggle()
        except:
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'attributes'):
                mgr.window.attributes('-fullscreen', True)
            else:
                mgr.window.state('zoomed')
    except Exception:
        pass

    # ── Map — NO aspect='equal', let it stretch wide ─────────────
    ax.set_facecolor(PANEL)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.axis('off')

    # Legend overlay (compact box, top-left corner)
    legend_handles = [
        mpatches.Patch(facecolor=CYAN, edgecolor=WHITE, linewidth=1, label='Depot'),
        mpatches.Patch(facecolor=BLUE, edgecolor=WHITE, linewidth=1, label='City'),
        plt.Line2D([0],[0], color=EDGE_HIGHWAY, lw=3, label='Highway x0.7'),
        plt.Line2D([0],[0], color=EDGE_NORMAL, lw=3, label='Normal'),
        plt.Line2D([0],[0], color=EDGE_MOUNTAIN, lw=3, label='Mountain x2'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=9, facecolor='#21262d', edgecolor=GRID,
              labelcolor=TXT, framealpha=0.9, borderpad=0.6)

    # Storm zones — thin dashed ring + small triangle
    for storm in problem.storm_zones:
        ring = mpatches.Circle(
            (storm.cx, storm.cy), storm.radius,
            facecolor='none', edgecolor='#ff444455',
            linewidth=0.8, linestyle=':', zorder=2)
        ax.add_patch(ring)
        ax.scatter(storm.cx, storm.cy, s=40, c='none',
                   edgecolors=RED, marker='^', linewidths=1, zorder=3)
        ax.text(storm.cx, storm.cy + storm.radius * 0.3,
                f'+{storm.penalty:.0f}', color=RED, fontsize=6,
                ha='center', va='bottom', alpha=0.55, zorder=3)

    # City nodes
    for city in cities:
        s  = DEPOT_S if city.is_depot else NODE_S
        fc = CYAN if city.is_depot else BLUE
        mk = 's' if city.is_depot else 'o'
        lw = 2.0 if city.is_depot else 0.8
        ax.scatter(city.x, city.y, s=s, c=fc, marker=mk,
                   edgecolors=WHITE, linewidths=lw, zorder=20)
        ax.text(city.x, city.y, str(city.id),
                color='#000' if city.is_depot else WHITE,
                fontsize=ID_FONT, fontweight='bold',
                ha='center', va='center', zorder=21)

    # START badge
    s0 = cities[0]
    ax.annotate('START', (s0.x, s0.y),
                textcoords='offset points', xytext=(0, -14),
                color=GREEN, fontsize=7, fontweight='bold',
                ha='center', va='top', zorder=22,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#238636',
                          edgecolor=GREEN, alpha=0.85))

    # ── Playback ─────────────────────────────────────────────────
    state = {'f': 0, 'play': False, 'tmr': None}
    dyn = []

    def _tour_edges(tour):
        """Return set of (min,max) edge pairs for a tour."""
        if tour is None: return set()
        full = list(tour) + [tour[0]]
        return {(min(full[i], full[i+1]), max(full[i], full[i+1]))
                for i in range(len(full)-1)}

    def draw(f):
        f = max(0, min(f, n_history - 1))
        state['f'] = f
        for o in dyn:
            try: o.remove()
            except: pass
        dyn.clear()

        snap = history[f]
        tour = snap.get('global_best_sol')
        bfit = snap.get('global_best_fit')

        if tour is not None and len(tour) == n:
            full = list(tour) + [tour[0]]
            for i in range(len(full) - 1):
                a, b = full[i], full[i + 1]
                ca, cb = cities[a], cities[b]
                raw   = problem.distance_matrix[a][b]
                tm    = problem.terrain_multiplier[a][b]
                sp    = problem._storm_penalty_for_edge(ca, cb)
                cost  = raw * tm + sp
                color = _tc(tm)

                dx, dy = cb.x - ca.x, cb.y - ca.y
                ln = np.hypot(dx, dy)
                if ln < 1e-6: continue

                # Thick visible arrow
                ann = ax.annotate(
                    '', xy=(cb.x, cb.y),
                    xytext=(ca.x, ca.y),
                    arrowprops=dict(
                        arrowstyle='-|>', color=color,
                        lw=EDGE_LW, mutation_scale=max(10, EDGE_LW*4),
                        shrinkA=NODE_S**0.5 / 2 + 3,
                        shrinkB=NODE_S**0.5 / 2 + 3,
                        connectionstyle='arc3,rad=0.04'),
                    zorder=8)
                dyn.append(ann)

                # Bold cost label
                mx, my = (ca.x + cb.x) / 2, (ca.y + cb.y) / 2
                lbl = f'{cost:.0f}'
                tl = _tl(tm)
                if tl: lbl += f' {tl}'
                if sp > 0: lbl += f' +{sp:.0f}S'

                t = ax.text(mx, my, lbl,
                            fontsize=COST_FONT, fontweight='bold',
                            color=color, ha='center', va='center',
                            zorder=15)
                dyn.append(t)

        # ── Build analysis text ──────────────────────────────────
        lines = []
        lines.append(f'--- Iteration {f+1}/{n_history} ---')
        lines.append('')

        # Global best
        if bfit is not None:
            lines.append(f'Global Best: {bfit:.1f}')
        
        # Check if global best improved this iteration
        if f > 0:
            prev_fit = history[f-1].get('global_best_fit')
            if prev_fit is not None and bfit is not None:
                if bfit < prev_fit:
                    delta = prev_fit - bfit
                    lines.append(f'>> IMPROVED by {delta:.1f} !!')
                else:
                    lines.append('(no improvement)')
        lines.append('')

        # Iter-best vs global-best
        iter_cost = snap.get('iter_best_cost')
        if iter_cost is not None:
            lines.append(f'Iter Best: {iter_cost:.1f}')
            if bfit is not None and iter_cost > bfit:
                lines.append(f'  (worse than global by {iter_cost - bfit:.1f})')
        lines.append('')

        # Ant colony statistics
        ant_mean = snap.get('ant_mean')
        if ant_mean is not None:
            lines.append('Ant Colony Stats:')
            lines.append(f'  Ants: {snap.get("n_ants", "?")}')
            lines.append(f'  Mean: {ant_mean:.1f}')
            lines.append(f'  Min:  {snap.get("ant_min", 0):.1f}')
            lines.append(f'  Max:  {snap.get("ant_max", 0):.1f}')
            lines.append(f'  Std:  {snap.get("ant_std", 0):.1f}')
        lines.append('')

        # Pheromone evaporation
        rho = snap.get('rho')
        if rho is not None:
            lines.append(f'Evaporation: {rho*100:.0f}%')
            lines.append(f'  (pheromone x {1-rho:.2f} each iter)')
        lines.append('')

        # Top pheromone edges
        pher = snap.get('pheromone')
        if pher is not None:
            # Get top-5 strongest pheromone edges
            tri = np.triu(pher, k=1)
            flat_idx = np.argsort(tri.ravel())[::-1][:5]
            rows, cols = np.unravel_index(flat_idx, tri.shape)
            lines.append('Strongest Pheromone:')
            for r, c in zip(rows, cols):
                lines.append(f'  {r}->{c}: {pher[r,c]:.2f}')
        lines.append('')

        # Edge changes vs previous iteration
        if f > 0:
            prev_tour = history[f-1].get('global_best_sol')
            cur_edges = _tour_edges(tour)
            prev_edges = _tour_edges(prev_tour)
            added = cur_edges - prev_edges
            removed = prev_edges - cur_edges
            if added or removed:
                lines.append('Route Changes:')
                for e in sorted(added):
                    lines.append(f'  + edge {e[0]}->{e[1]}')
                for e in sorted(removed):
                    lines.append(f'  - edge {e[0]}->{e[1]}')
                # Explain WHY
                if pher is not None and added:
                    lines.append('')
                    lines.append('Why changed:')
                    for e in sorted(added):
                        p_val = pher[e[0], e[1]]
                        lines.append(f'  {e[0]}->{e[1]} pher={p_val:.2f}')
                    lines.append('  High pheromone attracted')
                    lines.append('  ants to these edges')
            else:
                lines.append('Route: unchanged')

        info_str = '\n'.join(lines)
        ax_info.clear()
        ax_info.set_facecolor('#161b22')
        ax_info.axis('off')
        t = ax_info.text(0.05, 0.98, info_str,
                    fontsize=9, fontfamily='monospace',
                    color=TXT, ha='left', va='top',
                    transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#21262d',
                              edgecolor=GRID, alpha=0.95))
        dyn.append(t)

        cs = f'{bfit:.1f}' if bfit is not None else '?'
        ps = '||' if state['play'] else '>'
        fig.suptitle(
            f'{algo_name} -- TSP "{problem.name}"  |  '
            f'Iter {f+1}/{n_history}  |  Cost: {cs}  [{ps}]',
            color=TXT, fontsize=14, fontweight='bold', y=0.97)

        fig.canvas.draw_idle()

    # ── Controls ───────────────────────────────────────────────
    fig.text(0.015, 0.68, 
             'Controls:\n\n'
             '[ \u2190 ] Prev\n'
             '[ \u2192 ] Next\n'
             '[ Space ] Play / Pause\n'
             '[ Esc ] Exit\n\n'
             'Ctrl + Scroll : Zoom\n'
             'Mid Mouse Drag : Pan\n'
             'Home : Reset View',
             color=TXT, fontsize=9, ha='left', va='top', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#21262d', edgecolor=GRID, alpha=0.9))

    # Store original limits for Home reset
    _orig_xlim = (xlo, xhi)
    _orig_ylim = (ylo, yhi)
    _pan_state = {'active': False, 'x0': None, 'y0': None}

    def _prev():
        state['play'] = False; draw(state['f']-1)
    def _next():
        state['play'] = False; draw(state['f']+1)
    def _play():
        state['play'] = not state['play']
        draw(state['f'])
        if state['play']:
            _auto()
    def _auto():
        if not state['play']: return
        if state['f'] < n_history - 1:
            draw(state['f'] + 1)
            state['tmr'] = fig.canvas.new_timer(interval=interval)
            state['tmr'].add_callback(lambda: _auto())
            state['tmr'].single_shot = True
            state['tmr'].start()
        else:
            state['play'] = False
            draw(state['f'])

    def _on_scroll(event):
        if event.key != 'control' or event.inaxes != ax:
            return
        cur_xl, cur_xr = ax.get_xlim()
        cur_yb, cur_yt = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        scale = 0.8 if event.button == 'up' else 1.25
        ax.set_xlim(xdata - (xdata - cur_xl) * scale,
                    xdata + (cur_xr - xdata) * scale)
        ax.set_ylim(ydata - (ydata - cur_yb) * scale,
                    ydata + (cur_yt - ydata) * scale)
        fig.canvas.draw_idle()

    def _on_press(event):
        if event.button == 2 and event.inaxes == ax:
            _pan_state['active'] = True
            _pan_state['x0'] = event.xdata
            _pan_state['y0'] = event.ydata

    def _on_release(event):
        if event.button == 2:
            _pan_state['active'] = False

    def _on_motion(event):
        if not _pan_state['active'] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = _pan_state['x0'] - event.xdata
        dy = _pan_state['y0'] - event.ydata
        cur_xl, cur_xr = ax.get_xlim()
        cur_yb, cur_yt = ax.get_ylim()
        ax.set_xlim(cur_xl + dx, cur_xr + dx)
        ax.set_ylim(cur_yb + dy, cur_yt + dy)
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == 'escape':
            plt.close(fig)
        elif event.key == 'left':
            _prev()
        elif event.key == 'right':
            _next()
        elif event.key == ' ':
            _play()
        elif event.key == 'home':
            ax.set_xlim(*_orig_xlim)
            ax.set_ylim(*_orig_ylim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', _on_scroll)
    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    draw(0)

    if save_path:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            ani = FuncAnimation(fig, lambda f: draw(f) or [],
                                frames=n_history, interval=interval,
                                blit=False, repeat=False)
            ani.save(save_path, writer=PillowWriter(fps=max(1, 1000//interval)), dpi=100)
        except Exception as e:
            print(f'Warning: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)
