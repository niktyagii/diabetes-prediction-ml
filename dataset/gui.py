import tkinter as tk
from tkinter import ttk
import math
import random
import numpy as np

# ── ML libs (graceful fallback) ──────────────────────────────
try:
    import joblib
    ML_AVAILABLE = True
    try:
        model   = joblib.load("model.pkl")
        scaler  = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        MODEL_LOADED = True
    except Exception:
        MODEL_LOADED = False
except ImportError:
    ML_AVAILABLE  = False
    MODEL_LOADED  = False

# ── Palette ──────────────────────────────────────────────────
BG         = "#0a0f0a"
CARD       = "#111911"
BORDER     = "#1a2e1a"
GREEN      = "#00e676"
GREEN_DIM  = "#00b050"
GREEN_DARK = "#004d1a"
GREEN_GLOW = "#00ff88"
AMBER      = "#ffc107"
RED        = "#ff1744"
TEXT_DIM   = "#5a8a5a"
TEXT_MUTED = "#2d4d2d"
WHITE      = "#e8f5e9"

FM  = ("Courier New", 10)
FB  = ("Courier New", 11, "bold")
FS  = ("Courier New",  8)
F9  = ("Courier New",  9)
F9B = ("Courier New",  9, "bold")
F18 = ("Courier New", 18, "bold")
F20 = ("Courier New", 20, "bold")

# ── Fields ───────────────────────────────────────────────────
FIELDS = [
    ("Pregnancies",             "Times pregnant",          "0-17",    0,    17,   0),
    ("Glucose",                 "Plasma glucose level",    "50-200",  50,   200,  120),
    ("Blood Pressure",          "Diastolic (mm Hg)",       "40-130",  40,   130,  72),
    ("Skin Thickness",          "Tricep skin fold (mm)",   "0-100",   0,    100,  29),
    ("Insulin",                 "2-hr serum insulin",      "0-850",   0,    850,  125),
    ("BMI",                     "Body mass index",         "15-70",   15,   70,   32.0),
    ("Diabetes Pedigree Func.", "Hereditary score",        "0.08-2.5",0.08, 2.5,  0.5),
    ("Age",                     "Patient age (years)",     "21-90",   21,   90,   33),
]


# ════════════════════════════════════════════════════════════
#  ECG Canvas — live heartbeat line
# ════════════════════════════════════════════════════════════
class ECGCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._pts   = []
        self._x     = 0
        self._alive = True
        self.after(120, self._tick)

    def _ecg(self, x):
        x = x % 120
        if   30 <= x < 35: return -0.3
        elif 35 <= x < 38: return  1.0
        elif 38 <= x < 41: return -0.6
        elif 41 <= x < 48: return  0.4
        elif 48 <= x < 55: return  0.0
        return 0.05 * math.sin(x * 0.3)

    def _tick(self):
        try:
            if not self._alive or not self.winfo_exists():
                return
            w   = self.winfo_width()  or 500
            h   = self.winfo_height() or 50
            mid = h // 2
            amp = h * 0.38
            self._pts.append((self._x % w,
                               mid - int(self._ecg(self._x) * amp)))
            self._x += 2
            if len(self._pts) > w // 2 + 2:
                self._pts.pop(0)
            self.delete("ecg")
            if len(self._pts) >= 2:
                flat = [c for p in self._pts for c in p]
                self.create_line(*flat, fill=GREEN, width=1.5,
                                 tags="ecg", smooth=True)
            if self._pts:
                lx, ly = self._pts[-1]
                self.create_oval(lx-3, ly-3, lx+3, ly+3,
                                 fill=GREEN_GLOW, outline="", tags="ecg")
        except tk.TclError:
            return
        self.after(28, self._tick)

    def stop(self):
        self._alive = False


# ════════════════════════════════════════════════════════════
#  Animated health bar
# ════════════════════════════════════════════════════════════
class HealthBar(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._pct    = 0.0
        self._target = 0.0
        self.after(120, self._tick)

    def set_value(self, v):
        self._target = max(0.0, min(1.0, v))

    def _tick(self):
        try:
            if not self.winfo_exists():
                return
            self._pct += (self._target - self._pct) * 0.2
            self._redraw()
        except tk.TclError:
            return
        self.after(30, self._tick)

    def _redraw(self):
        try:
            self.delete("all")
            w = self.winfo_width()  or 160
            h = self.winfo_height() or 6
            self.create_rectangle(0, 0, w, h, fill="#1a2e1a", outline="")
            fw = int(w * self._pct)
            if fw > 1:
                self.create_rectangle(0, 0, fw, h, fill="#003311", outline="")
                self.create_rectangle(0, 1, fw, h-1, fill="#006622", outline="")
                self.create_rectangle(0, 1, fw, h-1, fill=GREEN, outline="")
            for i in range(1, 4):
                tx = w * i // 4
                self.create_line(tx, 0, tx, h, fill="#1f3f1f")
        except tk.TclError:
            pass


# ════════════════════════════════════════════════════════════
#  Glow button — uses Frame + Canvas label, no premature draw
# ════════════════════════════════════════════════════════════
class GlowButton(tk.Frame):
    def __init__(self, master, text, command,
                 color=GREEN, btn_width=160, btn_height=44, **kw):
        super().__init__(master, bg=BG,
                         width=btn_width, height=btn_height, **kw)
        self.pack_propagate(False)
        self._color    = color
        self._command  = command
        self._norm_bg  = "#0a1a0a"

        self._cv = tk.Canvas(self, bg=self._norm_bg,
                              highlightthickness=2,
                              highlightbackground=color,
                              cursor="hand2")
        self._cv.pack(fill="both", expand=True)

        self._lbl = tk.Label(self._cv, text=text, font=FB,
                              fg=color, bg=self._norm_bg, cursor="hand2")
        self._lbl.place(relx=0.5, rely=0.5, anchor="center")

        for widget in (self._cv, self._lbl):
            widget.bind("<Enter>",    self._enter)
            widget.bind("<Leave>",    self._leave)
            widget.bind("<Button-1>", self._click)

    def _enter(self, _=None):
        self._cv.config(bg=self._color)
        self._lbl.config(bg=self._color, fg=BG)

    def _leave(self, _=None):
        self._cv.config(bg=self._norm_bg)
        self._lbl.config(bg=self._norm_bg, fg=self._color)

    def _click(self, _=None):
        self._command()


# ════════════════════════════════════════════════════════════
#  Main application
# ════════════════════════════════════════════════════════════
class DiabetesApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Diabetes Prediction System")
        self.root.configure(bg=BG)
        self.root.geometry("1100x740")
        self.root.minsize(900, 660)

        self._vars = {}
        self._bars = {}

        self._build()
        # All canvas draws deferred until layout is fully settled
        self.root.after(200, self._initial_draw)

    # ── top-level layout ────────────────────────────────────
    def _build(self):
        self._build_header()
        self._hline(self.root)
        self._build_body()
        self._hline(self.root)
        self._build_footer()

    def _hline(self, parent):
        c = tk.Canvas(parent, height=2, bg=BG, highlightthickness=0)
        c.pack(fill="x")
        c.create_line(0, 1, 9999, 1, fill=BORDER, width=2)

    # ── HEADER ──────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG, height=88)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Logo block
        lf = tk.Frame(hdr, bg=BG)
        lf.pack(side="left", padx=18, pady=8)
        self._logo_cv = tk.Canvas(lf, width=44, height=44,
                                   bg=BG, highlightthickness=0)
        self._logo_cv.pack(side="left", padx=(0, 10))

        tf = tk.Frame(lf, bg=BG)
        tf.pack(side="left")
        tk.Label(tf, text="MEDDIAB\u00b7AI",
                 font=("Courier New", 19, "bold"),
                 fg=GREEN, bg=BG).pack(anchor="w")
        tk.Label(tf, text="Diabetes Risk Prediction System  v2.0",
                 font=FS, fg=TEXT_DIM, bg=BG).pack(anchor="w")

        # ECG strip
        ef = tk.Frame(hdr, bg=BG)
        ef.pack(side="left", fill="both", expand=True, padx=16)
        self._ecg = ECGCanvas(ef, bg=BG, highlightthickness=0, height=56)
        self._ecg.pack(fill="both", expand=True, pady=16)

        # Status dots
        sf = tk.Frame(hdr, bg=BG)
        sf.pack(side="right", padx=18)
        for label, ok in [("MODEL",  MODEL_LOADED),
                           ("ML LIB", ML_AVAILABLE),
                           ("ONLINE", True)]:
            row = tk.Frame(sf, bg=BG)
            row.pack(anchor="e", pady=1)
            dot = tk.Canvas(row, width=9, height=9,
                             bg=BG, highlightthickness=0)
            dot.pack(side="left", padx=(0, 4))
            dot.create_oval(1, 1, 8, 8,
                            fill=GREEN if ok else RED, outline="")
            tk.Label(row, text=label, font=FS,
                     fg=TEXT_DIM, bg=BG).pack(side="left")

    def _draw_dna(self):
        cv = self._logo_cv
        try:
            if not cv.winfo_exists():
                return
            for i in range(8):
                y  = 5 + i * 5
                x1 = 10 + int(9 * math.sin(i * 0.9))
                x2 = 34 - int(9 * math.sin(i * 0.9))
                c  = GREEN if i % 2 == 0 else GREEN_DIM
                cv.create_oval(x1-3, y-3, x1+3, y+3, fill=c, outline="")
                cv.create_oval(x2-3, y-3, x2+3, y+3, fill=c, outline="")
                cv.create_line(x1, y, x2, y, fill=TEXT_MUTED)
        except tk.TclError:
            pass

    # ── BODY ────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=8)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._build_inputs(left)

        right = tk.Frame(body, bg=BG, width=310)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)
        self._build_results(right)

    # ── INPUT PANEL ─────────────────────────────────────────
    def _build_inputs(self, parent):
        self._sec_head(parent, "  PATIENT DATA INPUT")
        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="both", expand=True, pady=2)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for idx, (name, hint, rng, lo, hi, default) in enumerate(FIELDS):
            self._field(grid, name, hint, rng, lo, hi, default,
                        row=idx // 2, col=idx % 2)

    def _field(self, grid, name, hint, rng, lo, hi, default, row, col):
        card = tk.Frame(grid, bg=CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=row, column=col, padx=5, pady=4, sticky="nsew")
        grid.rowconfigure(row, weight=1)

        hrow = tk.Frame(card, bg=CARD)
        hrow.pack(fill="x", padx=8, pady=(7, 1))
        tk.Label(hrow, text=name.upper(),
                 font=F9B, fg=GREEN, bg=CARD).pack(side="left")
        tk.Label(hrow, text=f"  [{rng}]",
                 font=FS, fg=TEXT_DIM, bg=CARD).pack(side="left")

        tk.Label(card, text=hint, font=FS,
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", padx=8)

        er = tk.Frame(card, bg=CARD)
        er.pack(fill="x", padx=8, pady=3)

        var = tk.StringVar(value=str(default))
        self._vars[name] = var

        tk.Entry(er, textvariable=var, width=11,
                 font=FM, fg=GREEN_GLOW, bg="#0d1a0d",
                 insertbackground=GREEN, relief="flat",
                 highlightthickness=1,
                 highlightbackground=GREEN_DARK,
                 highlightcolor=GREEN).pack(side="left")

        is_float = isinstance(default, float)
        lbl = tk.Label(er, text=str(default), width=8,
                       font=F9B, fg=GREEN_GLOW, bg=CARD)
        lbl.pack(side="right")

        bar = HealthBar(card, height=5, bg=CARD, highlightthickness=0)
        bar.pack(fill="x", padx=8, pady=(1, 6))
        self._bars[name] = bar
        bar.set_value((float(default) - lo) / max(hi - lo, 1))

        def _trace(*_, v=var, b=bar, lo=lo, hi=hi, lbl=lbl, fl=is_float):
            try:
                val = float(v.get())
                pct = (val - lo) / max(hi - lo, 1)
                b.set_value(pct)
                color = RED if pct > 0.75 else (AMBER if pct > 0.50 else GREEN_GLOW)
                lbl.config(fg=color,
                           text=f"{val:.1f}" if fl else str(int(val)))
            except Exception:
                lbl.config(fg=RED, text="?")
        var.trace_add("write", _trace)

    # ── RESULT PANEL ────────────────────────────────────────
    def _build_results(self, parent):
        self._sec_head(parent, "  ANALYSIS RESULT")

        fig_card = tk.Frame(parent, bg=CARD,
                            highlightthickness=1, highlightbackground=BORDER)
        fig_card.pack(fill="x", pady=(0, 6))
        self._fig_cv = tk.Canvas(fig_card, width=290, height=120,
                                  bg=CARD, highlightthickness=0)
        self._fig_cv.pack(pady=2)

        g_card = tk.Frame(parent, bg=CARD,
                          highlightthickness=1, highlightbackground=BORDER)
        g_card.pack(fill="x", pady=(0, 6))
        tk.Label(g_card, text="RISK PROBABILITY",
                 font=FS, fg=TEXT_DIM, bg=CARD).pack(pady=(6, 0))
        self._gauge_cv = tk.Canvas(g_card, width=290, height=125,
                                    bg=CARD, highlightthickness=0)
        self._gauge_cv.pack()

        log_card = tk.Frame(parent, bg=CARD,
                            highlightthickness=1, highlightbackground=BORDER)
        log_card.pack(fill="both", expand=True, pady=(0, 6))
        tk.Label(log_card, text="DIAGNOSTIC OUTPUT",
                 font=FS, fg=TEXT_DIM, bg=CARD).pack(pady=(6, 0))
        self._log = tk.Text(log_card, font=F9,
                            fg=GREEN, bg="#090f09",
                            relief="flat", state="disabled",
                            wrap="word", height=8,
                            highlightthickness=0, padx=8, pady=6)
        self._log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        ind_card = tk.Frame(parent, bg=CARD,
                            highlightthickness=1, highlightbackground=BORDER)
        ind_card.pack(fill="x")
        tk.Label(ind_card, text="RISK INDICATORS",
                 font=FS, fg=TEXT_DIM, bg=CARD).pack(pady=(6, 2))
        self._ind_row = tk.Frame(ind_card, bg=CARD)
        self._ind_row.pack(fill="x", padx=6, pady=(0, 6))

    def _sec_head(self, parent, text):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(0, 5))
        tk.Label(f, text=text, font=("Courier New", 9, "bold"),
                 fg=GREEN_DIM, bg=BG).pack(side="left")

    # ── FOOTER ──────────────────────────────────────────────
    def _build_footer(self):
        foot = tk.Frame(self.root, bg=BG, height=58)
        foot.pack(fill="x", padx=14, pady=6)
        foot.pack_propagate(False)

        tk.Label(foot,
                 text="  FOR CLINICAL DECISION SUPPORT ONLY \u2014 NOT A DIAGNOSTIC TOOL",
                 font=FS, fg=TEXT_MUTED, bg=BG).pack(side="left", anchor="w")

        bf = tk.Frame(foot, bg=BG)
        bf.pack(side="right", anchor="e")

        GlowButton(bf, "  RESET",   self._reset,
                   color=TEXT_DIM, btn_width=130, btn_height=38
                   ).pack(side="left", padx=(0, 10))
        GlowButton(bf, "  PREDICT", self._predict,
                   color=GREEN, btn_width=150, btn_height=38
                   ).pack(side="left")

    # ── DEFERRED INITIAL DRAW ───────────────────────────────
    def _initial_draw(self):
        self._draw_dna()
        self._draw_patient_idle()
        self._draw_gauge(0.0)
        self._draw_indicators({})
        self._log_write(
            "[ AWAITING INPUT ]\n\n"
            "Enter patient data and\n"
            "press PREDICT to run analysis."
        )

    # ── SAFE CANVAS HELPER ──────────────────────────────────
    def _safe(self, cv):
        try:
            return cv if cv.winfo_exists() else None
        except tk.TclError:
            return None

    # ── CANVAS DRAWS ────────────────────────────────────────
    def _draw_patient_idle(self):
        cv = self._safe(self._fig_cv)
        if not cv:
            return
        cv.delete("all")
        w, h, cx = 290, 120, 145
        for i in range(0, w, 28):
            cv.create_line(i, 0, i, h, fill=TEXT_MUTED)
        for j in range(0, h, 28):
            cv.create_line(0, j, w, j, fill=TEXT_MUTED)
        cv.create_oval(cx-16, 10, cx+16, 44,
                       fill="#1a2e1a", outline=GREEN_DIM, width=2)
        cv.create_rectangle(cx-20, 48, cx+20, 96,
                            fill="#142214", outline=GREEN_DIM, width=2)
        cv.create_line(cx-20, 54, cx-38, 86, fill=GREEN_DIM, width=2)
        cv.create_line(cx+20, 54, cx+38, 86, fill=GREEN_DIM, width=2)
        cv.create_line(cx, 58, cx, 84, fill=GREEN, width=3)
        cv.create_line(cx-11, 71, cx+11, 71, fill=GREEN, width=3)
        cv.create_text(cx, 110, text="PATIENT MODEL",
                       font=FS, fill=TEXT_DIM)

    def _draw_patient_result(self, diabetic, prob):
        cv = self._safe(self._fig_cv)
        if not cv:
            return
        cv.delete("all")
        w, cx, cy = 290, 145, 60
        col = RED if diabetic else GREEN
        for r in (52, 40):
            cv.create_oval(cx-r, cy-r, cx+r, cy+r,
                           fill="#200000" if diabetic else "#002000",
                           outline=col, width=1)
        cv.create_oval(cx-16, cy-34, cx+16, cy,
                       fill="#1a1010" if diabetic else "#101a10",
                       outline=col, width=2)
        cv.create_rectangle(cx-20, cy+4, cx+20, cy+50,
                            fill="#140a0a" if diabetic else "#0a140a",
                            outline=col, width=2)
        cv.create_line(cx-20, cy+10, cx-36, cy+42, fill=col, width=2)
        cv.create_line(cx+20, cy+10, cx+36, cy+42, fill=col, width=2)
        cv.create_text(cx, cy+28,
                       text="\u2715" if diabetic else "\u2713",
                       font=F20, fill=col)
        cv.create_text(cx, 112,
                       text="HIGH RISK DETECTED" if diabetic else "LOW RISK DETECTED",
                       font=FS, fill=col)

    def _draw_gauge(self, prob):
        cv = self._safe(self._gauge_cv)
        if not cv:
            return
        cv.delete("all")
        cx, cy, r = 145, 100, 72
        sa, arc = 210, 120
        for i in range(100):
            f   = i / 100
            ang = sa - f * arc
            c   = GREEN if f < 0.4 else (AMBER if f < 0.7 else RED)
            cv.create_arc(cx-r, cy-r, cx+r, cy+r,
                          start=ang, extent=-1.3,
                          style="arc", outline=c, width=7)
        if prob > 0.01:
            cv.create_arc(cx-r, cy-r, cx+r, cy+r,
                          start=sa, extent=-(prob * arc),
                          style="arc",
                          outline=(GREEN if prob < 0.4
                                   else AMBER if prob < 0.7 else RED),
                          width=11)
        ang_r = math.radians(sa - prob * arc)
        nx = cx + (r - 16) * math.cos(ang_r)
        ny = cy - (r - 16) * math.sin(ang_r)
        cv.create_line(cx, cy, nx, ny, fill=WHITE, width=2)
        cv.create_oval(cx-4, cy-4, cx+4, cy+4, fill=WHITE, outline="")
        cv.create_text(cx - r - 4, cy + 10, text="LOW",
                       font=FS, fill=GREEN, anchor="e")
        cv.create_text(cx + r + 4, cy + 10, text="HIGH",
                       font=FS, fill=RED, anchor="w")
        pct_col = GREEN if prob < 0.4 else (AMBER if prob < 0.7 else RED)
        cv.create_text(cx, cy + 18,
                       text=f"{prob*100:.1f}%", font=F18, fill=pct_col)

    def _draw_indicators(self, vals):
        for w in self._ind_row.winfo_children():
            w.destroy()
        checks = [
            ("Glucose",    vals.get("Glucose",        120), 50,  200, 140),
            ("BMI",        vals.get("BMI",            32.0), 15,  70,  25),
            ("Age",        vals.get("Age",             33),  21,  90,  45),
            ("BloodPres.", vals.get("Blood Pressure",  72),  40,  130, 90),
        ]
        for ci, (label, val, lo, hi, thresh) in enumerate(checks):
            f = tk.Frame(self._ind_row, bg=CARD)
            f.grid(row=0, column=ci, padx=3, sticky="ew")
            self._ind_row.columnconfigure(ci, weight=1)
            pct = (float(val) - lo) / max(hi - lo, 1)
            col = RED if float(val) > thresh else GREEN
            cv  = tk.Canvas(f, width=62, height=48,
                             bg=CARD, highlightthickness=0)
            cv.pack()
            ext = max(0, min(180, pct * 180))
            cv.create_arc(8, 4, 54, 42, start=180, extent=-180,
                          style="arc", outline=BORDER, width=5)
            cv.create_arc(8, 4, 54, 42, start=180, extent=-ext,
                          style="arc", outline=col, width=5)
            cv.create_text(31, 44, text=label, font=FS, fill=TEXT_DIM)

    # ── LOG ─────────────────────────────────────────────────
    def _log_write(self, text, color=None):
        try:
            self._log.config(state="normal")
            self._log.delete("1.0", "end")
            self._log.config(fg=color or GREEN)
            self._log.insert("end", text)
            self._log.config(state="disabled")
        except tk.TclError:
            pass

    # ── PREDICT ─────────────────────────────────────────────
    def _predict(self):
        vals, errors = {}, []
        for name, _, _, lo, hi, _ in FIELDS:
            raw = self._vars[name].get().strip()
            try:
                v = float(raw)
                if not (lo <= v <= hi):
                    errors.append(f"{name}: {v} not in [{lo}-{hi}]")
                vals[name] = v
            except ValueError:
                errors.append(f"{name}: invalid '{raw}'")

        if errors:
            self._log_write("[ INPUT ERROR ]\n\n" +
                            "\n".join(f"  x {e}" for e in errors), RED)
            return

        self._log_write("[ SCANNING ]\n\n  Preprocessing...\n  Running model...", AMBER)
        self.root.update_idletasks()

        order = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                 "Insulin", "BMI", "Diabetes Pedigree Func.", "Age"]
        X = np.array([[vals[k] for k in order]], dtype=float)

        if MODEL_LOADED:
            try:
                Xi = imputer.transform(X)
                Xs = scaler.transform(Xi)
                prob = float(model.predict_proba(Xs)[0][1])
                pred = int(model.predict(Xs)[0])
            except Exception:
                prob, pred = self._heuristic(vals)
        else:
            prob, pred = self._heuristic(vals)

        diabetic   = pred == 1
        risk_label = "HIGH" if prob > 0.7 else ("MODERATE" if prob > 0.4 else "LOW")
        col        = RED if diabetic else GREEN

        self._draw_patient_result(diabetic, prob)
        self._draw_gauge(prob)
        self._draw_indicators(vals)

        report = (
            f"[ PREDICTION COMPLETE ]\n\n"
            f"  RESULT   : {'DIABETIC' if diabetic else 'NON-DIABETIC'}\n"
            f"  RISK LVL : {risk_label}\n"
            f"  PROB     : {prob*100:.1f}%\n\n"
            f"  KEY FLAGS :\n"
            f"  Glucose  : {vals['Glucose']:.0f} mg/dL "
            f"{'[HIGH]' if vals['Glucose']>140 else '[OK]'}\n"
            f"  BMI      : {vals['BMI']:.1f} "
            f"{'[OBESE]' if vals['BMI']>30 else '[OK]'}\n"
            f"  Age      : {vals['Age']:.0f} yrs "
            f"{'[ELEVATED]' if vals['Age']>45 else '[OK]'}\n\n"
            f"  ACTION   : "
            f"{'Consult endocrinologist.' if diabetic else 'Maintain healthy lifestyle.'}"
        )
        self._log_write(report, col)

    def _heuristic(self, vals):
        s = 0.0
        if vals["Glucose"]                 > 140: s += 0.30
        if vals["BMI"]                     > 30:  s += 0.20
        if vals["Age"]                     > 45:  s += 0.15
        if vals["Diabetes Pedigree Func."] > 0.5: s += 0.15
        if vals["Blood Pressure"]          > 90:  s += 0.10
        if vals["Insulin"]                 > 200: s += 0.10
        s = max(0.04, min(0.96, s + random.uniform(-0.04, 0.04)))
        return s, int(s >= 0.5)

    # ── RESET ───────────────────────────────────────────────
    def _reset(self):
        for name, _, _, _, _, default in FIELDS:
            self._vars[name].set(str(default))
        self._draw_patient_idle()
        self._draw_gauge(0.0)
        self._draw_indicators({})
        self._log_write(
            "[ AWAITING INPUT ]\n\n"
            "Enter patient data and\n"
            "press PREDICT to run analysis."
        )


# ════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.configure(bg=BG)
    ttk.Style(root).theme_use("clam")
    app = DiabetesApp(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app._ecg.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()