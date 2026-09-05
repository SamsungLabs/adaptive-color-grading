'''
  Title::
dBadaCG.py
  Description::
Database manager for adaptive color grading interface
  Method::
User interface for interacting with gradeDict data structures of current projects and previous ones used for training. Calls adaCG.py color grading interface elements and allows for adaptive batch editing, provides visualizations.
  Inputs::
projPath - directory path of compatible images to import ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.exr', '.dpx')
projFile - previous project to continue editing ('.pkl' gradeDict data structure)
modelPath - previous project to use as training data ('.pkl' gradeDict data structure)
  Outputs::
project files, graded images, LUTs, model files for training
  Author::
Trevor D. Canham
  Correspondance::
tcanham@yorku.ca

Copyright (c) 2026 Samsung Electronics Co., Ltd.
'''

import colour as c
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import pickle
import copy
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3d projection

from PIL import Image, ImageDraw, ImageTk

# dBadaCG inherits its grading interface objects from adaCG
from adaCG import (REGIONS, BIT_DEPTHS, NBINS, adapt, colorEngine, cgEditor,
                   flatButton, flatToggle, defaultSet, defaultReg, imHist,
                   thumbDecode, imResize, newGrade)

IMTYPES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.exr', '.dpx')
HIST_MODES = ['none', 'unit', 'std', 'log', 'cdf']

# region threshold marker colors, darkest to lightest
REG_COLORS = {'darkest': '#9b59b6', 'dark': '#5dade2', 'light': '#2ecc71', 'lightest': '#e74c3c'}

# top level gradeDict keys that are not image grades. 'set' and 'reg' show up at the top
# level of pkls written by earlier adaCG builds, which stashed the live editor state there.
RESERVED = ('root', 'set', 'reg')


def gradeSig(grade):
    # cheap identity for a grade, used to key the rendered thumb / gscale caches
    return '%r|%r' % (grade.get('set'), grade.get('reg'))


def sameImage(a, b):
    # decide whether two grades under the same key describe the same image, so a harmless
    # re-add can be told from a real filename collision. compares stored data only, no field
    # is added to the grade and no source image is read.
    for f in ('thumb', 'hist'):
        x, y = a.get(f), b.get(f)
        if x is None or y is None:
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return False
        return bool(np.allclose(x, y)) if x.dtype.kind == 'f' else bool(np.array_equal(x, y))
    return False # nothing to compare on, treat it as a conflict so the user is told


def parseTime(s):
    s = (s or '').strip()
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def fmtTime(t):
    return t.strftime('%Y-%m-%d %H:%M') if isinstance(t, datetime) else ''


def gradeTime(grade):
    # sort key. older pkls stored the datetime.now function itself rather than calling it
    t = grade.get('time')
    return t if isinstance(t, datetime) else datetime.min


def shortPath(p, n):
    p = str(p)
    return p if len(p) <= n else '...' + p[-(n - 3):]


class mergeResult():
    # outcome of any operation that folds grades into a db, so conflicts can be reported
    # rather than silently dropped

    def __init__(self):
        self.added = []
        self.dupes = []       # same key, same image. already present, nothing to do
        self.conflicts = []   # same key, different image. a real filename collision
        self.unreadable = []

    def summary(self):
        bits = ['%d added' % len(self.added)]
        if self.dupes:
            bits.append('%d already present' % len(self.dupes))
        if self.conflicts:
            bits.append('%d name conflict(s) skipped' % len(self.conflicts))
        if self.unreadable:
            bits.append('%d unreadable' % len(self.unreadable))
        return ', '.join(bits)


# ---------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------

class gradeDB():
    # thin wrapper over a gradeDict, whose layout is adaCG's and is not extended here:
    #   {'root': <path>, '<key>': {'set','reg','hist','thumb','labels','time'}}
    # keys are paths relative to 'root' (a bare file name for anything sitting directly in
    # the root, which is what adaCG writes). a key that resolves outside the root is stored
    # absolute, and os.path.join below returns it unchanged.

    def __init__(self, path=None, loadsImages=True):
        self.gd = {'root': ''}
        self.path = path
        self.loadsImages = loadsImages # the train db never reads source images
        if path is not None and os.path.exists(path):
            self.load(path)

    # ---- io ----

    def load(self, path):
        with open(path, "rb") as file:
            gd = pickle.load(file)
        if not isinstance(gd, dict):
            raise ValueError('%s does not hold a gradeDict' % path)
        self.gd = gd
        self.gd.setdefault('root', os.path.dirname(path))
        self.path = path
        self.normalize()
        return self

    def save(self, path=None):
        path = path or self.path
        if path is None:
            raise ValueError('gradeDB.save: no path set')
        with open(path, "wb") as file:
            pickle.dump(self.gd, file)
        self.path = path
        return path

    def normalize(self):
        # bring an older pkl up to the current grade layout without adding anything to it
        for k in ('set', 'reg'):
            self.gd.pop(k, None) # leftover top level editor state, not a grade
        for k in self.keys():
            grade = self.gd[k]
            grade.setdefault('set', defaultSet())
            grade.setdefault('reg', defaultReg())
            grade.setdefault('hist', None)
            grade.setdefault('thumb', None)
            grade.setdefault('labels', [])
            grade.setdefault('time', datetime.now())

    # ---- access ----

    @property
    def root(self):
        return self.gd.get('root', '')

    @root.setter
    def root(self, value):
        self.gd['root'] = value

    def keys(self):
        return sorted([k for k in self.gd.keys()
                       if k not in RESERVED and isinstance(self.gd[k], dict)])

    def grade(self, key):
        return self.gd[key]

    def name(self, key):
        return os.path.basename(key)

    def fullPath(self, key):
        # os.path.join returns the second argument unchanged when it is absolute, so this
        # resolves both root relative and absolute keys
        return os.path.join(self.root, key)

    def has(self, key):
        return key in self.gd and key not in RESERVED and isinstance(self.gd[key], dict)

    def add(self, key, grade):
        self.gd[key] = grade

    def remove(self, key):
        if self.has(key):
            del self.gd[key]

    def nGrades(self):
        return len(self.keys())

    def histBins(self):
        for k in self.keys():
            h = self.gd[k].get('hist')
            if h is not None:
                return int(np.asarray(h).size)
        return None

    def keyFor(self, dirPath, name):
        # a path relative to the root where possible, absolute when the file sits outside it
        full = os.path.join(dirPath, name)
        if not self.root:
            return name
        rel = os.path.relpath(full, self.root)
        return full if rel.startswith('..') else rel

    # ---- ingest ----

    def offer(self, key, grade, res):
        # single place where a candidate grade meets an existing one, so every ingest path
        # reports conflicts the same way
        if self.has(key):
            (res.dupes if sameImage(self.gd[key], grade) else res.conflicts).append(key)
            return False
        self.add(key, grade)
        res.added.append(key)
        return True

    def ingestDirectory(self, dirPath, ftypes=IMTYPES, nBins=NBINS, histMode='std',
                        thumbScale=0.05, progress=None, onImage=None):
        # onImage hands the caller the image that was just read, so a preview can be built
        # from it rather than reading every file a second time
        res = mergeResult()
        names = [f for f in sorted(os.listdir(dirPath)) if f.lower().endswith(tuple(ftypes))]
        for i, nm in enumerate(names):
            key = self.keyFor(dirPath, nm)
            if self.has(key): # already known, do not re-read it just to compare
                res.dupes.append(key)
                continue
            if progress is not None:
                progress(i + 1, len(names), nm)
            try:
                fullIm = c.read_image(os.path.join(dirPath, nm)) # float32 in 0-1 range
            except Exception as e:
                print('dBadaCG: could not read %s (%s)' % (nm, e))
                res.unreadable.append(key)
                continue
            if self.offer(key, newGrade(fullIm, nBins, histMode, thumbScale), res) \
               and onImage is not None:
                onImage(key, fullIm)
        return res

    def mergeFrom(self, path):
        # "add from file": fold grades out of another gradeDict pkl into this db
        with open(path, "rb") as file:
            other = pickle.load(file)
        res = mergeResult()
        for k in sorted(other.keys()):
            if k in RESERVED or not isinstance(other[k], dict):
                continue
            grade = copy.deepcopy(other[k])
            # same fill in as normalize(), so a grade merged out of an older pkl still has
            # a grade to render a ramp and thresholds from
            grade.setdefault('set', defaultSet())
            grade.setdefault('reg', defaultReg())
            grade.setdefault('labels', [])
            grade.setdefault('time', datetime.now())
            self.offer(k, grade, res)
        return res

    def copyGradesFrom(self, srcDb, keys):
        res = mergeResult()
        for k in keys:
            self.offer(k, copy.deepcopy(srcDb.grade(k)), res)
        return res

    def rehistogram(self, nBins, histMode, progress=None):
        # a bit depth change invalidates every stored histogram, adapt refuses to mix them
        ok, missing = [], []
        keys = self.keys()
        for i, k in enumerate(keys):
            p = self.fullPath(k)
            if not os.path.exists(p):
                missing.append(k)
                continue
            if progress is not None:
                progress(i + 1, len(keys), self.name(k))
            try:
                fullIm = c.read_image(p)
            except Exception as e:
                print('dBadaCG: could not read %s (%s)' % (p, e))
                missing.append(k)
                continue
            self.gd[k]['hist'] = imHist(fullIm, nBins, histMode)
            ok.append(k)
        return ok, missing

    # ---- filtering ----

    def filterKeys(self, labels='', root=''):
        wants = [s.strip().lower() for s in labels.split(',') if s.strip()]
        out = []
        for k in self.keys():
            grade = self.gd[k]
            if wants:
                have = [str(l).lower() for l in grade.get('labels', [])]
                if not any(w in h for w in wants for h in have):
                    continue
            if root and root.lower() not in str(self.root).lower() \
               and root.lower() not in k.lower():
                continue
            out.append(k)
        return out


# ---------------------------------------------------------------------------
# filter bar
# ---------------------------------------------------------------------------

class filterBar(tk.Frame):

    def __init__(self, master, fields, onChange=None, **kw):
        super().__init__(master, bg='gray15', **kw)
        self.onChange = onChange
        self.vars = {}
        tk.Label(self, text='filter by:', bg='gray15', fg='gray80').pack(side=tk.LEFT, padx=(4, 2))
        for f in fields:
            tk.Label(self, text=f, bg='gray15', fg='gray60').pack(side=tk.LEFT, padx=(6, 1))
            v = tk.StringVar(self, '')
            v.trace_add('write', lambda *a: self.fire())
            tk.Entry(self, textvariable=v, width=12, bg='gray25', fg='gray99',
                     insertbackground='gray99', relief=tk.FLAT).pack(side=tk.LEFT)
            self.vars[f] = v
        flatButton(self, text='clear', command=self.clear).pack(side=tk.LEFT, padx=4)

    def fire(self):
        if self.onChange is not None:
            self.onChange()

    def clear(self):
        for v in self.vars.values():
            v.set('')

    def spec(self):
        g = lambda k: self.vars[k].get() if k in self.vars else ''
        return {'labels': g('labels'), 'root': g('root')}


# ---------------------------------------------------------------------------
# grade view
# ---------------------------------------------------------------------------

class gradeView(tk.Frame):
    # canvas backed list of gradeDict grades, as a details list or a gallery of tiles.
    # only the rows actually inside the viewport are drawn, so redraw cost is set by the
    # size of the window rather than the size of the database.

    ROWH = 58
    TILEW = 140
    TILEH = 164
    THUMB = 48
    GSW, GSH = 110, 14

    def __init__(self, master, db, onSelect=None, onActivate=None, thumbFn=None, gscaleFn=None, **kw):
        super().__init__(master, bg='gray9', **kw)
        self.db = db
        self.onSelect = onSelect
        self.onActivate = onActivate
        self.thumbFn = thumbFn      # host supplies (db, key, box) -> graded PIL thumb
        self.gscaleFn = gscaleFn    # host supplies (db, key, w, h) -> graded PIL ramp

        self.keys = []
        self.selection = set()
        self.anchor = None
        self.mode = 'list'
        self.imRefs = []            # PhotoImage refs, tkinter will not hold them for us

        self.canvas = tk.Canvas(self, bg='gray12', highlightthickness=0)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.onScrollBar)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas.bind('<Button-1>', self.onClick)
        self.canvas.bind('<Double-Button-1>', self.onDouble)
        self.canvas.bind('<Configure>', lambda e: self.redraw())
        self.canvas.bind('<MouseWheel>', self.onWheel)

    # ---- state ----

    def setKeys(self, keys):
        self.keys = list(keys)
        self.selection &= set(self.keys) # a filtered out grade cannot stay selected
        self.redraw()

    def setMode(self, mode):
        self.mode = mode
        self.redraw()

    def selectedKeys(self):
        return [k for k in self.keys if k in self.selection] # display order

    def selectOnly(self, key):
        self.selection = {key}
        self.anchor = key
        self.redraw()

    # ---- scrolling. culling means the view has to be redrawn as it scrolls ----

    def onScrollBar(self, *args):
        self.canvas.yview(*args)
        self.redraw()

    def onWheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), 'units')
        self.redraw()

    # ---- hit testing ----

    def nCols(self):
        return max(1, int(max(self.canvas.winfo_width(), self.TILEW) // self.TILEW))

    def indexAt(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.mode == 'list':
            idx = int(y // self.ROWH)
        else:
            idx = int(y // self.TILEH) * self.nCols() + int(x // self.TILEW)
        return idx if 0 <= idx < len(self.keys) else None

    def onClick(self, event):
        self.canvas.focus_set()
        idx = self.indexAt(event)
        if idx is None:
            return
        key = self.keys[idx]
        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        if shift and self.anchor in self.keys:
            a = self.keys.index(self.anchor)
            lo, hi = sorted((a, idx))
            self.selection = set(self.keys[lo:hi + 1])
        elif ctrl:
            self.selection.symmetric_difference_update({key})
            self.anchor = key
        else:
            self.selection = {key}
            self.anchor = key
        self.redraw()
        if self.onSelect is not None:
            self.onSelect(self.db, self.selectedKeys())

    def onDouble(self, event):
        idx = self.indexAt(event)
        if idx is not None and self.onActivate is not None:
            self.onActivate(self.db, self.keys[idx])

    # ---- image helpers ----

    def tkThumb(self, key, box):
        pil = self.thumbFn(self.db, key, box) if self.thumbFn is not None else None
        if pil is None:
            return None
        tkim = ImageTk.PhotoImage(pil)
        self.imRefs.append(tkim)
        return tkim

    def tkGscale(self, key, w, h):
        pil = self.gscaleFn(self.db, key, w, h) if self.gscaleFn is not None else None
        if pil is None:
            return None
        tkim = ImageTk.PhotoImage(pil)
        self.imRefs.append(tkim)
        return tkim

    # ---- draw ----

    def visibleRange(self, span, perRow):
        # index window to draw, padded by one row either side so partial rows are present
        top = self.canvas.canvasy(0)
        bot = top + max(self.canvas.winfo_height(), 1)
        i0 = max(0, (int(top // span) - 1) * perRow)
        i1 = min(len(self.keys), (int(bot // span) + 2) * perRow)
        return i0, i1

    def redraw(self):
        self.canvas.delete('all')
        self.imRefs = []
        if self.mode == 'list':
            self.drawList()
        else:
            self.drawGallery()

    def drawList(self):
        w = max(self.canvas.winfo_width(), 320)
        # scrollregion covers every grade, only the visible slice is actually drawn
        self.canvas.configure(scrollregion=(0, 0, w, max(len(self.keys) * self.ROWH, 1)))
        i0, i1 = self.visibleRange(self.ROWH, 1)
        for i in range(i0, i1):
            key = self.keys[i]
            y = i * self.ROWH
            grade = self.db.grade(key)
            if key in self.selection:
                self.canvas.create_rectangle(0, y, w, y + self.ROWH, fill='gray30', outline='')
            self.canvas.create_line(0, y + self.ROWH, w, y + self.ROWH, fill='gray20')
            x = 4
            tkim = self.tkThumb(key, self.THUMB)
            if tkim is not None:
                self.canvas.create_image(x, y + self.ROWH / 2, anchor=tk.W, image=tkim)
            x += self.THUMB + 8
            self.canvas.create_text(x, y + 16, anchor=tk.W, text=self.db.name(key),
                                    fill='gray95', font=('TkDefaultFont', 9))
            self.canvas.create_text(x, y + 34, anchor=tk.W, text=shortPath(key, 46),
                                    fill='gray55', font=('TkDefaultFont', 7))
            x += 230
            tkim = self.tkGscale(key, self.GSW, self.GSH)
            if tkim is not None:
                self.canvas.create_image(x, y + self.ROWH / 2, anchor=tk.W, image=tkim)
            x += self.GSW + 10
            self.canvas.create_text(x, y + self.ROWH / 2, anchor=tk.W,
                                    text=', '.join(str(l) for l in grade.get('labels', [])),
                                    fill='gray75', font=('TkDefaultFont', 8))
            x += 120
            self.canvas.create_text(x, y + self.ROWH / 2, anchor=tk.W, text=fmtTime(grade.get('time')),
                                    fill='gray60', font=('TkDefaultFont', 8))
            if grade.get('hist') is None:
                # without a histogram a grade can neither train nor be predicted on
                self.canvas.create_text(w - 8, y + self.ROWH / 2, anchor=tk.E, text='no hist',
                                        fill='#c0504d', font=('TkDefaultFont', 7))

    def drawGallery(self):
        nc = self.nCols()
        rows = int(np.ceil(len(self.keys) / float(nc))) if self.keys else 1
        self.canvas.configure(scrollregion=(0, 0, nc * self.TILEW, max(rows * self.TILEH, 1)))
        i0, i1 = self.visibleRange(self.TILEH, nc)
        for i in range(i0, i1):
            key = self.keys[i]
            grade = self.db.grade(key)
            cx = (i % nc) * self.TILEW
            cy = (i // nc) * self.TILEH
            if key in self.selection:
                self.canvas.create_rectangle(cx + 2, cy + 2, cx + self.TILEW - 2,
                                             cy + self.TILEH - 2, fill='gray30', outline='')
            box = self.TILEW - 24
            yy = cy + 8
            tkim = self.tkThumb(key, box)
            if tkim is not None:
                self.canvas.create_image(cx + self.TILEW / 2, yy + box / 2, image=tkim)
            yy += box + 4
            tkim = self.tkGscale(key, box, 10)
            if tkim is not None:
                self.canvas.create_image(cx + self.TILEW / 2, yy, image=tkim)
            yy += 14
            self.canvas.create_text(cx + self.TILEW / 2, yy, text=shortPath(self.db.name(key), 20),
                                    fill='gray95', font=('TkDefaultFont', 8))
            yy += 14
            self.canvas.create_text(cx + self.TILEW / 2, yy,
                                    text=', '.join(str(l) for l in grade.get('labels', []))[:24],
                                    fill='gray75', font=('TkDefaultFont', 7))
            yy += 12
            self.canvas.create_text(cx + self.TILEW / 2, yy, text=fmtTime(grade.get('time')),
                                    fill='gray60', font=('TkDefaultFont', 7))


# ---------------------------------------------------------------------------
# db pane
# ---------------------------------------------------------------------------

class dbPane(tk.Frame):

    def __init__(self, master, app, db, title, buttons, side='left', feedsEditor=True, **kw):
        super().__init__(master, bg='gray9', **kw)
        self.app = app
        self.db = db
        self.title = title

        head = tk.Frame(self, bg='gray15')
        head.pack(fill=tk.X)
        self.titleVar = tk.StringVar(self, title)
        tk.Label(head, textvariable=self.titleVar, bg='gray15', fg='gray99',
                 font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=6, pady=2)
        self.modeVar = tk.StringVar(self, 'list')
        flatToggle(head, self.modeVar, ('list', 'gallery'), width=6, bg='gray15',
                   command=lambda v: self.setMode()).pack(side=tk.RIGHT, padx=2)
        self.sortVar = tk.StringVar(self, 'name')
        srt = tk.OptionMenu(head, self.sortVar, 'name', 'date', command=lambda v: self.refresh())
        srt.configure(bg='gray20', fg='gray90', highlightthickness=0, relief=tk.FLAT)
        srt.pack(side=tk.RIGHT, padx=4)
        tk.Label(head, text='sort by:', bg='gray15', fg='gray60').pack(side=tk.RIGHT)

        self.filters = filterBar(self, ['labels', 'root'], onChange=self.refresh)
        self.filters.pack(fill=tk.X)

        body = tk.Frame(self, bg='gray9')
        body.pack(expand=True, fill=tk.BOTH)

        btnCol = tk.Frame(body, bg='gray9')
        btnCol.pack(side=(tk.LEFT if side == 'left' else tk.RIGHT), fill=tk.Y, padx=2)
        for label, cmd in buttons:
            flatButton(btnCol, text=label, command=cmd,
                       wraplength=88, width=12).pack(pady=2, fill=tk.X)

        self.view = gradeView(body, db,
                               onSelect=app.onSelect,
                               onActivate=(app.loadGrade if feedsEditor else None),
                               thumbFn=app.gradedThumb, gscaleFn=app.gscaleStrip)
        self.view.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.statusVar = tk.StringVar(self, '')
        tk.Label(self, textvariable=self.statusVar, bg='gray12', fg='gray60',
                 anchor=tk.W).pack(fill=tk.X)

    def setMode(self):
        self.view.setMode(self.modeVar.get())

    def setStatus(self, text):
        self.statusVar.set(text)

    def sortKeys(self, keys):
        if self.sortVar.get() == 'date':
            return sorted(keys, key=lambda k: gradeTime(self.db.grade(k)))
        return sorted(keys, key=lambda k: k.lower())

    def refresh(self):
        keys = self.sortKeys(self.db.filterKeys(**self.filters.spec()))
        self.view.setKeys(keys)
        self.titleVar.set('%s  [%s]' % (self.title, shortPath(self.db.root, 40)))
        self.setStatus('%d of %d grades   |   %s' %
                       (len(keys), self.db.nGrades(), shortPath(self.db.path or '(unsaved)', 60)))

    def selectedKeys(self):
        return self.view.selectedKeys()


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

class dBadaCGapp(tk.Tk):

    UNDO_DEPTH = 64
    GS_SAMPLES = 32   # the gscale ramp is built at this width, then resized up for display

    def __init__(self, projectPath=None, trainPath=None, ftype='.tif', projectFile=None):

        super().__init__()
        self.title("dBadaCG: adaCG grade database")
        self.configure(bg='gray9')

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.geometry('%dx%d' % (int(self.width * 0.9), int(self.height * 0.9)))

        # settings
        self.ftype = ftype
        self.nBins = tk.IntVar(self, NBINS)
        self.histMode = tk.StringVar(self, 'std')
        self.nNeighbors = tk.IntVar(self, 16)
        self.previewDiv = tk.IntVar(self, 6) # preview image height = screen width / this
        self.thumbScale = 0.05
        self.batchMode = tk.StringVar(self, 'individual')
        self.statusVar = tk.StringVar(self, 'ready')

        # engine and state. everything the UI construction touches has to exist first.
        self.nodes = 17
        self.ce = colorEngine(self.nodes)
        self.editor = None
        self.curKey = None            # project key in the editor. train never loads.
        self.previews = {}            # project key -> downscaled float32 preview
        self.previewW = None          # width the held previews were built at
        self.thumbCache = {}          # (key, box, gradeSig) -> PIL
        self.gsCache = {}             # (gradeSig, w, h) -> PIL, shared across grades
        self.undoStack = []
        self.redoStack = []
        self.dirty = False
        self.refreshJob = None
        self.suppressEdits = False    # set while the editor is being driven programmatically

        self.projDb = gradeDB(loadsImages=True)
        self.trainDb = gradeDB(loadsImages=False)

        self.buildMenu()
        self.buildLayout()
        self.protocol('WM_DELETE_WINDOW', self.onClose)
        self.refreshPanes()

        # load after the window is up, so the loading indicator is actually visible
        if trainPath:
            self.after(10, lambda: self.openModel(trainPath, quiet=True))
        if projectPath:
            self.after(20, lambda: self.loadProjectDir(projectPath, projectFile))

    # ---- menu -----------------------------------------------------------

    def buildMenu(self):
        menubar = tk.Menu(self)

        fileM = tk.Menu(menubar, tearoff=0)
        fileM.add_command(label='Undo', accelerator='Ctrl+Z', command=self.undo)
        fileM.add_command(label='Redo', accelerator='Ctrl+Y', command=self.redo)
        fileM.add_separator()
        fileM.add_command(label='Open Project...', command=self.openProject)
        fileM.add_command(label='Save Project', accelerator='Ctrl+S', command=self.saveProject)
        fileM.add_command(label='Save Project As...', command=self.saveProjectAs)
        fileM.add_separator()
        fileM.add_command(label='Save Image...', command=self.saveIm)
        fileM.add_command(label='Save Batch...', command=self.saveBatch)
        fileM.add_command(label='Save Lut...', command=self.saveLUT)
        fileM.add_separator()
        # the "model" on disk is the curated train gradeDict. the KNN refits from it in no
        # time, so there is nothing gained by pickling a fitted estimator.
        fileM.add_command(label='Save Model', command=self.saveModel)
        fileM.add_command(label='Save Model As...', command=self.saveModelAs)
        fileM.add_command(label='Load Model...', command=self.loadModel)
        fileM.add_separator()
        fileM.add_command(label='Quit', command=self.onClose)
        menubar.add_cascade(label='File', menu=fileM)

        projM = tk.Menu(menubar, tearoff=0)
        projM.add_command(label='Import from Directory...', command=self.importProjectDir)
        projM.add_command(label='Import from Grade File...', command=self.importProjectFile)
        projM.add_separator()
        projM.add_command(label='Remove Selected from Project', command=self.removeFromProject)
        projM.add_command(label='Add Selected to Train', command=self.addToTrain)
        projM.add_command(label='Label Selected...', command=lambda: self.labelSelected(self.projPane))
        menubar.add_cascade(label='Project', menu=projM)

        trainM = tk.Menu(menubar, tearoff=0)
        trainM.add_command(label='Import from File...', command=self.importTrainFile)
        trainM.add_command(label='Remove Selected from Train', command=self.removeFromTrain)
        trainM.add_separator()
        trainM.add_command(label='Apply Model to Selected', command=self.applyModel)
        trainM.add_command(label='Label Selected...', command=lambda: self.labelSelected(self.trainPane))
        menubar.add_cascade(label='Train', menu=trainM)

        setM = tk.Menu(menubar, tearoff=0)
        depthM = tk.Menu(setM, tearoff=0)
        for bits, bins in sorted(BIT_DEPTHS.items()):
            depthM.add_radiobutton(label='%d bit  (%d bins)' % (bits, bins), value=bins,
                                   variable=self.nBins)
        setM.add_cascade(label='histogram bit depth', menu=depthM)
        handM = tk.Menu(setM, tearoff=0)
        for m in HIST_MODES:
            handM.add_radiobutton(label=m, value=m, variable=self.histMode)
        setM.add_cascade(label='histogram handling', menu=handM)
        # no train equivalent: train grades have no resolvable source image. change the bit
        # depth, rehistogram the project, then re-add to train.
        setM.add_command(label='Rehistogram Project', command=self.rehistogramProject)
        setM.add_separator()
        # preview size is fixed at the previewDiv default above. changing it forces a full
        # re-read of the project, so it is not offered from the menu. uncomment to restore.
        # sizeM = tk.Menu(setM, tearoff=0)
        # for d, lab in ((16, 'small'), (8, 'medium'), (6, 'default'), (4, 'large'), (2, 'very large')):
        #     sizeM.add_radiobutton(label='%s (screen/%d)' % (lab, d), value=d,
        #                           variable=self.previewDiv, command=self.reloadPreviews)
        # setM.add_cascade(label='preview size', menu=sizeM)
        knnM = tk.Menu(setM, tearoff=0)
        for k in (1, 2, 4, 8, 16, 32):
            knnM.add_radiobutton(label='k = %d' % k, value=k, variable=self.nNeighbors)
        setM.add_cascade(label='adapt neighbors', menu=knnM)
        menubar.add_cascade(label='Settings', menu=setM)

        self.config(menu=menubar)
        self.bind_all('<Control-z>', lambda e: self.undo())
        self.bind_all('<Control-y>', lambda e: self.redo())
        self.bind_all('<Control-s>', lambda e: self.saveProject())

    # ---- layout ---------------------------------------------------------

    def buildLayout(self):

        outer = tk.PanedWindow(self, orient=tk.VERTICAL, bg='gray9', sashwidth=6,
                               sashrelief=tk.RAISED)
        outer.pack(expand=True, fill=tk.BOTH)

        dbRow = tk.Frame(outer, bg='gray9')
        outer.add(dbRow, stretch='always')

        self.projPane = dbPane(
            dbRow, self, self.projDb, 'Project',
            [('import', self.importProjectDir),
             ('remove from project', self.removeFromProject),
             ('add to train', self.addToTrain),
             ('save project', self.saveProject)],
            side='left', feedsEditor=True)
        self.projPane.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2, pady=2)

        self.trainPane = dbPane(
            dbRow, self, self.trainDb, 'Train',
            [('import', self.importTrainFile),
             ('remove from train', self.removeFromTrain),
             ('apply model to selected', self.applyModel),
             ('save model', self.saveModel)],
            side='right', feedsEditor=False)
        self.trainPane.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2, pady=2)

        edRow = tk.Frame(outer, bg='gray9')
        outer.add(edRow, stretch='always')

        btnCol = tk.Frame(edRow, bg='gray9')
        btnCol.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        for label, cmd in (('reset', self.resetGrade),
                           ('prev im', self.prevIm),
                           ('next im', self.nextIm),
                           ('save im', self.saveIm),
                           ('save lut', self.saveLUT)):
            flatButton(btnCol, text=label, command=cmd,
                       wraplength=88, width=12).pack(pady=3, fill=tk.X)

        # the visualization pane claims its slice before the editor
        self.vizPane = tk.Frame(edRow, bg='gray9', width=420)
        self.vizPane.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.vizPane.pack_propagate(False)

        # the grading controls themselves come straight out of adaCG
        self.editor = cgEditor(edRow, ce=self.ce, nodes=self.nodes,
                               gradWidth=int(self.width / 8), onChange=self.onGradeChange)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # batch switch, above the chroma control
        sw = tk.Frame(self.editor.control_frame, bg='gray9')
        sw.pack(pady=4, before=self.editor.canvas_wheel)
        tk.Label(sw, text='color offset applies to:', bg='gray9', fg='gray70').pack()
        row = tk.Frame(sw, bg='gray9')
        row.pack()
        flatToggle(row, self.batchMode, ('individual', 'batch'),
                   width=9, bg='gray9').pack()

        self.buildViz()

        tk.Label(self, textvariable=self.statusVar, bg='gray15', fg='gray70',
                 anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    def buildViz(self):
        self.fig = Figure(figsize=(4.2, 4.6), facecolor='#171717')
        self.axHist = self.fig.add_subplot(2, 1, 1, facecolor='#171717')
        self.axLut = self.fig.add_subplot(2, 1, 2, projection='3d', facecolor='#171717')
        self.fig.subplots_adjust(left=0.16, right=0.96, top=0.94, bottom=0.06, hspace=0.35)
        self.vizCanvas = FigureCanvasTkAgg(self.fig, master=self.vizPane)
        self.vizCanvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        self.drawViz()

    def drawViz(self):
        self.axHist.clear()
        self.axHist.set_facecolor('#171717')
        grade = self.curGrade()
        h = None if grade is None else grade.get('hist')
        if h is not None:
            h = np.asarray(h).reshape(-1)
            self.axHist.fill_between(np.linspace(0, 1, h.size), 0, h, color='#9a9a9a', linewidth=0)
        else:
            self.axHist.text(0.5, 0.5, 'no histogram', color='#808080', ha='center', va='center')
        self.axHist.set_xlim(0, 1)
        for reg in REGIONS:
            piv = self.editor.regDict[reg][0]
            self.axHist.axvline(piv, color=REG_COLORS[reg], linestyle='--', linewidth=1.4,
                                label='%s %.3f' % (reg, piv))
        self.axHist.set_title('tonescale region thresholds', color='#d9d9d9', fontsize=8)
        self.axHist.tick_params(colors='#9a9a9a', labelsize=6)
        leg = self.axHist.legend(fontsize=10, facecolor='#222222', edgecolor='none',
                                 loc='upper right')
        for t in leg.get_texts(): # labelcolor= is not on older matplotlib
            t.set_color('#d9d9d9')

        # 3D LUT. reuses the LUT the editor already built, and plots few enough points that
        # the redraw does not stall the event loop.
        self.axLut.clear()
        self.axLut.set_facecolor('#171717')
        lutOut = self.editor.lastLut
        if lutOut is not None:
            try:
                tbl = np.clip(np.asarray(lutOut.table), 0, 1)
                s = max(1, tbl.shape[0] // 5) # ~5 nodes per axis on screen
                pts = tbl[::s, ::s, ::s, :].reshape(-1, 3)
                self.axLut.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts, s=10, depthshade=False)
            except Exception as e:
                print('dBadaCG: LUT plot failed (%s)' % e)
        self.axLut.set_title('3D LUT', color='#d9d9d9', fontsize=8)
        try:
            for ax in (self.axLut.xaxis, self.axLut.yaxis, self.axLut.zaxis):
                ax.set_pane_color((0.09, 0.09, 0.09, 1.0))
        except Exception:
            pass # the pane color api has moved between matplotlib versions
        self.axLut.tick_params(colors='#808080', labelsize=5)
        self.vizCanvas.draw_idle()

    # ---- rendering grades ---------------------------------------------

    def lutFor(self, grade):
        # when a grade carries the grade the editor is currently showing, the editor has
        # already built that LUT. this is exactly the batch edit case, where every touched
        # grade shares one grade. colorEngine is untouched either way.
        if self.editor is not None and self.editor.lastLut is not None \
           and gradeSig(grade) == gradeSig(self.editor.gradeDict()):
            return self.editor.lastLut
        lutOut, _ = self.ce.apply(grade)
        return lutOut

    def gradedThumb(self, db, key, box):
        # the stored thumb, downsampled to its display size and then put through the
        # grade's own grade
        grade = db.grade(key)
        ck = (id(db), key, box, gradeSig(grade))
        hit = self.thumbCache.get(ck)
        if hit is not None:
            return hit
        pil = thumbDecode(grade.get('thumb'))
        if pil is None:
            return None
        pil = pil.convert('RGB')
        pil.thumbnail((box, box)) # downsample before the LUT, not after
        try:
            arr = np.asarray(pil, dtype=np.float32) / np.float32(255)
            out = np.clip(self.lutFor(grade).apply(arr), 0, 1)
            pil = Image.fromarray(np.multiply(out, 255).astype(np.uint8))
        except Exception as e:
            print('dBadaCG: thumb grade failed on %s (%s)' % (key, e))
        if len(self.thumbCache) > 400:
            self.thumbCache.clear()
        self.thumbCache[ck] = pil
        return pil

    def gscaleStrip(self, db, key, w, h):
        # the black to white ramp under this grade's grade, so the list doubles as a visual
        # index of what each grade does, marked with that grade's four region thresholds.
        # keyed by the grade alone, so grades sharing a grade share one strip, and any pivot
        # change (a manual drag or an applied model) is a new key and so redraws by itself.
        grade = db.grade(key)
        ck = (gradeSig(grade), w, h)
        hit = self.gsCache.get(ck)
        if hit is not None:
            return hit
        try:
            n = min(self.GS_SAMPLES, max(2, w)) # ramp built small, then resized up
            grad = np.repeat(np.reshape(np.linspace(0, 1, n, dtype=np.float32), (1, n, 1)), 3, axis=2)
            out = np.clip(self.lutFor(grade).apply(grad), 0, 1)
            pil = Image.fromarray(np.multiply(out, 255).astype(np.uint8)).resize((w, h))
            # region threshold markers, drawn after the resize so they stay one pixel wide.
            # same colors as the visualization pane, darkest to lightest.
            drawer = ImageDraw.Draw(pil)
            for reg in REGIONS:
                piv = float(grade['reg'][reg][0])
                x = max(0, min(w - 1, int(round(piv * (w - 1))))) # a pivot can land outside 0-1
                drawer.line([(x, 0), (x, h - 1)], fill=REG_COLORS[reg], width=1)
        except Exception as e:
            print('dBadaCG: gscale failed on %s (%s)' % (key, e))
            return None
        if len(self.gsCache) > 400:
            self.gsCache.clear()
        self.gsCache[ck] = pil
        return pil

    # ---- editor plumbing ------------------------------------------------

    def curGrade(self):
        if self.curKey is None or not self.projDb.has(self.curKey):
            return None
        return self.projDb.grade(self.curKey)

    def previewWidth(self):
        return int(self.width / max(1, self.previewDiv.get()))

    def loadGrade(self, db, key):
        # only project grades reach the editor. the train window is a curation list and
        # never touches source images.
        if db is not self.projDb or not db.has(key):
            return
        self.curKey = key
        grade = db.grade(key)
        self.suppressEdits = True # setGrade / setImage must not read back as a user edit
        try:
            self.editor.setGrade(grade['set'], grade['reg'], refresh=False)
            im = self.previews.get(key)
            if im is None:
                self.status('no preview loaded for %s' % key)
            self.editor.setImage(im)
        finally:
            self.suppressEdits = False
        self.title('dBadaCG%s: %s' % (' *' if self.dirty else '', db.fullPath(key)))
        self.drawViz()

    def onSelect(self, db, keys):
        if db is self.projDb:
            if keys:
                self.loadGrade(db, keys[0])
            self.status('%d selected in project' % len(keys))
        else:
            # train selection deliberately does not drive the editor
            self.status('%d selected in train' % len(keys))

    def onGradeChange(self, setDict, regDict, field='set'):
        # field is the half of the grade the editor actually changed. only that half is
        # written, so a colour edit cannot carry the loaded grade's tonescale onto the rest
        # of the selection.
        if self.suppressEdits or self.curKey is None:
            return
        targets = [self.curKey]
        if field == 'set' and self.batchMode.get() == 'batch':
            # the switch reads "color offset applies to:", so it governs offsets only.
            # a pivot drag is always individual.
            sel = self.projPane.selectedKeys()
            targets = sel if sel else [self.curKey]
        self.pushUndo(targets)
        for k in targets:
            if not self.projDb.has(k):
                continue
            grade = self.projDb.grade(k)
            grade[field] = copy.deepcopy(setDict if field == 'set' else regDict)
            grade['time'] = datetime.now()
        self.markDirty()
        # the chroma box fires this on every motion event, so the list redraw and the
        # matplotlib replot are coalesced instead of run per pixel of drag
        self.scheduleRefresh()
        if len(targets) > 1:
            self.status('batch color offset applied to %d grades (unsaved)' % len(targets))

    def resetGrade(self):
        # adaCGapp's reset: a*b* colour offsets back to zero for every region. the tonescale
        # pivots and falloffs are left alone, which is what adaCG has always done.
        if self.curKey is None:
            return self.status('no project grade loaded')
        n = len(self.projPane.selectedKeys()) if self.batchMode.get() == 'batch' else 1
        self.editor.reset() # fires onGradeChange, so batch mode applies as it does for a drag
        self.status('reset color offsets on %d grade(s) (unsaved)' % max(1, n))

    def scheduleRefresh(self, delay=200):
        if self.refreshJob is not None:
            self.after_cancel(self.refreshJob)
        self.refreshJob = self.after(delay, self.runScheduledRefresh)

    def runScheduledRefresh(self):
        self.refreshJob = None
        self.refreshPanes()
        self.drawViz()

    def refreshPanes(self):
        self.projPane.refresh()
        self.trainPane.refresh()

    def clearRenderCaches(self):
        self.thumbCache.clear()
        self.gsCache.clear()

    def status(self, msg):
        self.statusVar.set(msg)
        self.update_idletasks()

    def markDirty(self):
        self.dirty = True
        base = 'dBadaCG *'
        self.title('%s: %s' % (base, self.projDb.fullPath(self.curKey))
                   if self.curKey else base)

    def markClean(self):
        self.dirty = False
        self.title('dBadaCG: %s' % (self.projDb.path or ''))

    # ---- undo / redo ----------------------------------------------------

    def snapshot(self, keys):
        return [(k, copy.deepcopy(self.projDb.grade(k)['set']),
                 copy.deepcopy(self.projDb.grade(k)['reg']))
                for k in keys if self.projDb.has(k)]

    def pushUndo(self, keys):
        self.undoStack.append(self.snapshot(keys))
        if len(self.undoStack) > self.UNDO_DEPTH:
            self.undoStack.pop(0)
        self.redoStack = []

    def restore(self, items):
        back = self.snapshot([k for k, _, _ in items])
        for k, s, r in items:
            if self.projDb.has(k):
                self.projDb.grade(k)['set'] = s
                self.projDb.grade(k)['reg'] = r
        if self.curKey is not None and self.projDb.has(self.curKey):
            grade = self.projDb.grade(self.curKey)
            self.suppressEdits = True
            try:
                self.editor.setGrade(grade['set'], grade['reg'])
            finally:
                self.suppressEdits = False
        self.markDirty()
        self.refreshPanes()
        self.drawViz()
        return back

    def undo(self):
        if not self.undoStack:
            return self.status('nothing to undo')
        self.redoStack.append(self.restore(self.undoStack.pop()))
        self.status('undo')

    def redo(self):
        if not self.redoStack:
            return self.status('nothing to redo')
        self.undoStack.append(self.restore(self.redoStack.pop()))
        self.status('redo')

    # ---- conflict reporting ---------------------------------------------

    def reportMerge(self, res, what):
        # a filename collision means two different images want the same key. the grade is
        # left alone and the user is told, rather than one silently replacing the other.
        self.status('%s: %s' % (what, res.summary()))
        if res.conflicts:
            shown = '\n'.join('  ' + k for k in res.conflicts[:12])
            more = '' if len(res.conflicts) <= 12 else '\n  ... and %d more' % (len(res.conflicts) - 12)
            messagebox.showwarning(
                'filename conflict',
                '%d incoming grade(s) use a name that already exists in the %s database, '
                'but describe a different image. they were skipped and nothing was '
                'overwritten.\n\n%s%s\n\nrename the files, or keep them in separate '
                'projects.' % (len(res.conflicts), what, shown, more))

    # ---- loading --------------------------------------------------------

    def loadProgress(self, i, n, name):
        self.projPane.setStatus('loading %d of %d project images   |   %s' % (i, n, name))
        self.update_idletasks()

    def capturePreview(self, key, fullIm):
        # called back during an ingest, while the image is already in hand
        self.previews[key] = imResize(fullIm, self.previewW)

    def loadPreviews(self):
        # every project image is held at display resolution, so stepping between images and
        # batch edits are instant. this is the cost paid once, up front.
        w = self.previewWidth()
        if self.previewW != w: # a preview size change invalidates everything held
            self.previews = {}
            self.previewW = w
        keys = self.projDb.keys()
        todo = [k for k in keys if k not in self.previews] # ingest already built the rest
        for i, k in enumerate(todo):
            p = self.projDb.fullPath(k)
            self.projPane.setStatus('loading %d of %d project images   |   %s'
                                    % (i + 1, len(todo), self.projDb.name(k)))
            self.update_idletasks()
            if not os.path.exists(p):
                continue
            try:
                self.previews[k] = imResize(c.read_image(p), w)
            except Exception as e:
                print('dBadaCG: could not read %s (%s)' % (p, e))
        missing = len(keys) - len(self.previews)
        self.projPane.refresh()
        self.status('%d of %d project previews loaded at %d px%s'
                    % (len(self.previews), len(keys), w,
                       '  (%d source images not found)' % missing if missing else ''))

    def reloadPreviews(self):
        if self.projDb.nGrades():
            self.loadPreviews()
            if self.curKey is not None:
                self.loadGrade(self.projDb, self.curKey)

    def scanTypes(self):
        return (self.ftype,) if self.ftype else IMTYPES

    # ---- project commands ----------------------------------------------

    def loadProjectDir(self, d, pkl=None):
        # sets the project root from a directory and scans it. grades come from pkl, which
        # defaults to grade.pkl in that directory. anything in the directory without a grade
        # gets one. nothing is written back until Save Project.
        d = os.path.normpath(d)
        pkl = os.path.normpath(pkl) if pkl else os.path.join(d, 'grade.pkl')
        self.projDb.gd = {'root': d}
        self.projDb.path = pkl
        if os.path.exists(pkl):
            self.projDb.load(pkl)
            self.projDb.root = d # the pkl may have been moved, the directory opened wins
        self.curKey = None
        self.previews = {}
        self.previewW = self.previewWidth()
        self.clearRenderCaches()
        self.undoStack, self.redoStack = [], []
        self.projPane.setStatus('scanning %s ...' % d)
        self.update_idletasks()
        res = self.projDb.ingestDirectory(d, self.scanTypes(), self.nBins.get(),
                                          self.histMode.get(), self.thumbScale,
                                          progress=self.loadProgress,
                                          onImage=self.capturePreview)
        self.markClean()
        self.projPane.refresh()
        self.loadPreviews()
        self.reportMerge(res, 'project')

    def importProjectDir(self):
        d = filedialog.askdirectory(title='import images into project')
        if not d:
            return
        if not self.projDb.nGrades() and not self.projDb.root:
            self.projDb.root = os.path.normpath(d) # the first import defines the root
        res = self.projDb.ingestDirectory(d, self.scanTypes(), self.nBins.get(),
                                          self.histMode.get(), self.thumbScale,
                                          progress=self.loadProgress,
                                          onImage=self.capturePreview)
        if res.added:
            self.markDirty()
        self.loadPreviews()
        self.reportMerge(res, 'project')

    def importProjectFile(self):
        p = filedialog.askopenfilename(title='import into project from grade file',
                                       filetypes=[('grade pickle', '*.pkl'), ('all', '*.*')])
        if not p:
            return
        res = self.projDb.mergeFrom(p)
        if res.added:
            self.markDirty()
        self.clearRenderCaches()
        self.loadPreviews()
        self.reportMerge(res, 'project')

    def removeFromProject(self):
        keys = self.projPane.selectedKeys()
        if not keys:
            return self.status('nothing selected in project')
        if not messagebox.askyesno('remove from project',
                                   'remove %d grade(s) from the project gradeDict?\n\n'
                                   'image files on disk are not touched, and the project '
                                   'file is not changed until Save Project.' % len(keys)):
            return
        for k in keys:
            self.projDb.remove(k)
            self.previews.pop(k, None)
            if self.curKey == k:
                self.curKey = None
        self.markDirty()
        self.refreshPanes()
        self.status('removed %d grades from project (unsaved)' % len(keys))

    def labelSelected(self, pane):
        keys = pane.selectedKeys()
        if not keys:
            return self.status('nothing selected')
        s = simpledialog.askstring('label', 'comma separated labels for %d grade(s):' % len(keys),
                                   parent=self)
        if s is None:
            return
        labs = [t.strip() for t in s.split(',') if t.strip()]
        for k in keys:
            pane.db.grade(k)['labels'] = list(labs)
        self.markDirty()
        pane.refresh() # labels change no grade, so no thumb or gscale is invalidated
        self.status('labelled %d grades (unsaved)' % len(keys))

    # ---- train commands -------------------------------------------------

    def addToTrain(self):
        keys = self.projPane.selectedKeys()
        if not keys:
            return self.status('nothing selected in project')
        res = self.trainDb.copyGradesFrom(self.projDb, keys)
        if not self.trainDb.root:
            self.trainDb.root = self.projDb.root
        self.trainPane.refresh()
        self.reportMerge(res, 'train')

    def importTrainFile(self):
        p = filedialog.askopenfilename(title='import into train from grade file',
                                       filetypes=[('grade pickle', '*.pkl'), ('all', '*.*')])
        if not p:
            return
        res = self.trainDb.mergeFrom(p)
        self.trainPane.refresh()
        self.reportMerge(res, 'train')

    def removeFromTrain(self):
        keys = self.trainPane.selectedKeys()
        if not keys:
            return self.status('nothing selected in train')
        for k in keys:
            self.trainDb.remove(k)
        self.trainPane.refresh()
        self.status('removed %d grades from train' % len(keys))

    # ---- model ----------------------------------------------------------

    def buildModel(self):
        # refit from the curated train set. cheap enough that there is no reason to hold or
        # persist a fitted estimator.
        keys = [k for k in (self.trainPane.view.keys or self.trainDb.keys())
                if self.trainDb.grade(k).get('hist') is not None]
        if not keys:
            messagebox.showerror('model', 'no train grades with a histogram.')
            return None
        bins = int(np.asarray(self.trainDb.grade(keys[0])['hist']).size)
        try:
            return adapt(self.trainDb.gd, keys, nBins=bins, nNeighbors=self.nNeighbors.get())
        except Exception as e:
            messagebox.showerror('model', str(e))
            return None

    def applyModel(self):
        keys = self.projPane.selectedKeys()
        if not keys:
            return self.status('select the project grades to apply the model to')
        keys = [k for k in keys if self.projDb.grade(k).get('hist') is not None]
        if not keys:
            return self.status('none of the selected project grades have a histogram')
        model = self.buildModel()
        if model is None:
            return
        bad = [k for k in keys if np.asarray(self.projDb.grade(k)['hist']).size != model.nBins]
        if bad:
            messagebox.showerror('apply model',
                                 '%d selected grade(s) have a histogram bin count that does '
                                 'not match the model (%d bins).\n\nrun Settings > '
                                 'Rehistogram Project first.' % (len(bad), model.nBins))
            return
        if not messagebox.askyesno(
                'apply model',
                'set predicted region pivots on %d selected project grade(s)?\n\n'
                'only the four pivots change. a*b* offsets and falloffs are left as they are.\n\n'
                'the change stays in this session only. the project file is not updated '
                'unless you choose Save Project.' % len(keys)):
            return
        self.pushUndo(keys)
        H = np.vstack([np.asarray(self.projDb.grade(k)['hist']).reshape(1, -1) for k in keys])
        pred = model.query(H)
        for i, k in enumerate(keys):
            reg = self.projDb.grade(k)['reg']
            for j, r in enumerate(model.regions):
                reg[r][0] = float(pred[i, j]) # prediction sets the pivot only
            self.projDb.grade(k)['time'] = datetime.now()
        self.markDirty()
        if self.curKey in keys:
            self.loadGrade(self.projDb, self.curKey)
        self.refreshPanes()
        self.drawViz()
        self.status('model applied to %d project grades (unsaved)' % len(keys))

    def saveModel(self):
        # the model file is the curated train gradeDict, in the same layout as any other
        if self.trainDb.path is None:
            return self.saveModelAs()
        if not self.trainDb.nGrades():
            return self.status('train set is empty, nothing to save')
        self.trainDb.save()
        self.trainPane.refresh()
        self.status('saved model (%d train grades) to %s'
                    % (self.trainDb.nGrades(), self.trainDb.path))

    def saveModelAs(self):
        if not self.trainDb.nGrades():
            return self.status('train set is empty, nothing to save')
        p = filedialog.asksaveasfilename(title='save model as', defaultextension='.pkl',
                                         initialfile='model.pkl',
                                         filetypes=[('grade pickle', '*.pkl')])
        if not p:
            return
        self.trainDb.save(p)
        self.trainPane.refresh()
        self.status('saved model (%d train grades) to %s' % (self.trainDb.nGrades(), p))

    def loadModel(self):
        p = filedialog.askopenfilename(title='load model',
                                       filetypes=[('grade pickle', '*.pkl'), ('all', '*.*')])
        if p:
            self.openModel(p)

    def openModel(self, p, quiet=False):
        if not os.path.exists(p):
            self.trainDb.path = p # remembered for Save Model
            return
        try:
            self.trainDb.load(p)
        except Exception as e:
            if not quiet:
                messagebox.showerror('load model', str(e))
            return
        self.trainPane.db = self.trainDb
        self.trainPane.view.db = self.trainDb
        self.clearRenderCaches()
        self.trainPane.refresh()
        self.status('loaded model (%d train grades) from %s' % (self.trainDb.nGrades(), p))

    # ---- settings -------------------------------------------------------

    def rehistogramProject(self):
        nBins, mode = self.nBins.get(), self.histMode.get()
        if not messagebox.askyesno('rehistogram',
                                   'recompute %d project histograms at %d bins, handling '
                                   '"%s"?\n\nevery source image is read again. the project '
                                   'file is not updated unless you choose Save Project.'
                                   % (self.projDb.nGrades(), nBins, mode)):
            return
        ok, missing = self.projDb.rehistogram(nBins, mode, progress=self.loadProgress)
        if ok:
            self.markDirty()
        self.projPane.refresh()
        self.drawViz()
        msg = 'rehistogrammed %d grades at %d bins (unsaved)' % (len(ok), nBins)
        if missing:
            msg += '  (%d skipped, source image not found)' % len(missing)
        self.status(msg)

    # ---- project io -----------------------------------------------------

    def openProject(self):
        if not self.confirmDiscard('open a different project'):
            return
        p = filedialog.askopenfilename(title='open project',
                                       filetypes=[('grade pickle', '*.pkl'), ('all', '*.*')])
        if p:
            self.loadProjectFile(p)

    def loadProjectFile(self, p):
        # a project is a gradeDict pkl, and its 'root' says where the images live
        self.projDb.load(p)
        self.projPane.db = self.projDb
        self.projPane.view.db = self.projDb
        self.curKey = None
        self.previews = {} # keys belong to the old project, they cannot be carried over
        self.previewW = self.previewWidth()
        self.clearRenderCaches()
        self.undoStack, self.redoStack = [], []
        self.markClean()
        self.projPane.refresh()
        self.loadPreviews()
        self.status('opened project %s (%d grades)' % (p, self.projDb.nGrades()))

    def saveProject(self):
        if self.projDb.path is None:
            return self.saveProjectAs()
        if not self.projDb.nGrades():
            return self.status('project is empty, nothing to save')
        self.projDb.save()
        self.markClean()
        self.projPane.refresh()
        self.status('saved project to %s' % self.projDb.path)

    def saveProjectAs(self):
        p = filedialog.asksaveasfilename(title='save project as', defaultextension='.pkl',
                                         initialfile='grade.pkl',
                                         filetypes=[('grade pickle', '*.pkl')])
        if not p:
            return
        self.projDb.save(p)
        self.markClean()
        self.projPane.refresh()
        self.status('saved project to %s' % p)

    def confirmDiscard(self, what):
        if not self.dirty:
            return True
        return messagebox.askyesno('unsaved changes',
                                   'the project has unsaved changes.\n\n'
                                   'discard them and %s?' % what)

    def onClose(self):
        if self.dirty and not messagebox.askyesno(
                'unsaved changes', 'the project has unsaved changes.\n\nquit anyway?'):
            return
        self.destroy()

    # ---- image io -------------------------------------------------------

    def prevIm(self):
        self.stepIm(-1)

    def nextIm(self):
        self.stepIm(1)

    def stepIm(self, d):
        # steps through the project pane in filtered display order
        keys = self.projPane.view.keys
        if not keys:
            return
        if self.curKey in keys:
            i = (keys.index(self.curKey) + d) % len(keys)
        else:
            i = 0 if d > 0 else len(keys) - 1 # nothing loaded, enter at the near end
        self.projPane.view.selectOnly(keys[i])
        self.loadGrade(self.projDb, keys[i])

    def saveIm(self):
        if self.curGrade() is None:
            return self.status('no project grade loaded')
        src = self.projDb.fullPath(self.curKey)
        if not os.path.exists(src):
            return self.status('source image missing: %s' % src)
        p = filedialog.asksaveasfilename(
            title='save image', defaultextension='.jpg',
            initialfile=os.path.splitext(self.projDb.name(self.curKey))[0] + '_graded.jpg',
            filetypes=[('jpeg', '*.jpg'), ('tiff', '*.tif'), ('png', '*.png')])
        if not p:
            return
        self.status('writing %s ...' % p)
        lutOut = self.editor.update_image() # current LUT object
        fullIm = c.read_image(src) # float32 in 0-1 range
        c.write_image(np.clip(lutOut.apply(fullIm), 0, 1), p, bit_depth='uint8')
        self.status('wrote %s' % p)

    def saveBatch(self):
        # writes every selected project grade through its own grade
        keys = self.projPane.selectedKeys() or self.projPane.view.keys
        if not keys:
            return self.status('nothing to batch save')
        d = filedialog.askdirectory(title='save batch to directory')
        if not d:
            return
        if not messagebox.askyesno('save batch', 'write %d graded images to\n%s ?' % (len(keys), d)):
            return
        n = 0
        for i, k in enumerate(keys):
            src = self.projDb.fullPath(k)
            if not os.path.exists(src):
                print('dBadaCG: skipping missing %s' % src)
                continue
            self.status('batch %d/%d  %s' % (i + 1, len(keys), self.projDb.name(k)))
            try:
                lutOut, _ = self.ce.apply(self.projDb.grade(k)) # each grade uses its own grade
                fullIm = c.read_image(src)
                out = os.path.join(d, os.path.splitext(self.projDb.name(k))[0] + '_graded.jpg')
                c.write_image(np.clip(lutOut.apply(fullIm), 0, 1), out, bit_depth='uint8')
                n += 1
            except Exception as e:
                print('dBadaCG: batch save failed on %s (%s)' % (src, e))
        self.status('batch saved %d of %d images to %s' % (n, len(keys), d))

    def saveLUT(self):
        if self.curGrade() is None:
            return self.status('no project grade loaded')
        p = filedialog.asksaveasfilename(title='save lut', defaultextension='.cube',
                                         initialfile='lut.cube', filetypes=[('cube LUT', '*.cube')])
        if not p:
            return
        c.write_LUT(self.editor.update_image(), p)
        self.status('wrote %s' % p)


if __name__ == "__main__":

    # bundled demo. paths are resolved against this file rather than the working directory,
    # so the demo runs from wherever the repository was cloned to.
    here = os.path.dirname(os.path.abspath(__file__))

    # project root, and the gradeDict pkl the grades come from. the pkl sits in the root
    # here, but it does not have to.
    projPath = os.path.join(here, 'demoProject')
    projFile = os.path.join(here, 'demoProject', 'demoProject.pkl')
    modelPath = os.path.join(here, 'demoProject', 'trainDemo8b.pkl')
    app = dBadaCGapp(projPath, modelPath, ftype='.tif', projectFile=projFile)
    app.mainloop()
