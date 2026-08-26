# raw grader app
import colour as c
import numpy as np
import tkinter as tk
import os
from PIL import Image,ImageTk,ImageDraw
import pdb
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import io
from datetime import datetime

# region names, in the order the color engine applies them and the order adapt predicts them
REGIONS = ['darkest','dark','light','lightest']

# histogram bit depth options exposed in the dBadaCG settings menu
BIT_DEPTHS = {8: 256, 10: 1024, 12: 4096, 14: 16384, 16: 65536}

# default histogram bit depth. one place, so every record built by either app agrees.
# adapt refuses to mix bin counts, so a record made here has to match the model.
NBINS = BIT_DEPTHS[8]


def defaultSet():
    # init a*b* offsets
    return {'darkest': [0,0], 'dark': [0,0], 'light': [0,0], 'lightest': [0,0]}


def defaultReg():
    # [pivot, falloff] for each region
    return {'darkest': [0.1,-10], 'dark': [0.4,-5], 'light': [0.5,5], 'lightest': [0.9,10]}


def imHist(im, nBins=NBINS, mode='std'):
    # luminance histogram used as the adapt model input feature
    # NOTE: the bin range is pinned to 0-1 so that histograms from different images share bin edges.
    # without a fixed range np.histogram picks per-image edges and the feature vectors are not comparable.
    imXYZ = c.RGB_to_XYZ(im, c.models.RGB_COLOURSPACE_DISPLAY_P3, apply_cctf_decoding=True) # convert to CIE XYZ
    hist, edge = np.histogram(imXYZ[:,:,1].flatten(), nBins, range=(0.0,1.0))
    return histNorm(hist, mode)


def histNorm(hist, mode='std'):
    # histogram handling, selectable from the settings menu
    hist = np.asarray(hist, dtype=np.float64).reshape(-1)
    if mode == 'none':
        return hist
    elif mode == 'unit': # sum to one, removes image resolution from the feature
        s = hist.sum()
        return hist / s if s > 0 else hist
    elif mode == 'log': # compress the huge dynamic range of bin counts, then sum to one
        hist = np.log1p(hist)
        s = hist.sum()
        return hist / s if s > 0 else hist
    elif mode == 'cdf': # cumulative, makes the feature robust to bin-level noise
        s = hist.sum()
        return np.cumsum(hist) / s if s > 0 else np.cumsum(hist)
    else: # 'std', scale by standard deviation without centering
        scaler = StandardScaler(with_mean=False)
        return scaler.fit_transform(hist.reshape(-1,1)).reshape(-1)


def imThumb(im, scale=0.05, quality=85):
    # encode a small jpeg preview and hand back the raw byte stream.
    # stored in the gradeDict rather than a decoded array to keep the pkl small
    thumb = Image.fromarray(np.multiply(np.clip(im,0,1),255).astype(np.uint8))
    thumb = thumb.resize((max(1,int(im.shape[1]*scale)), max(1,int(im.shape[0]*scale))))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=quality)  # Set JPEG compression quality here
    jpeg_bytes = buffer.getvalue()
    return np.frombuffer(jpeg_bytes, dtype=np.uint8)


def thumbDecode(buffThumb):
    # decode a stored thumb byte stream back to a PIL image
    if buffThumb is None:
        return None
    try:
        return Image.open(io.BytesIO(np.asarray(buffThumb, dtype=np.uint8).tobytes()))
    except Exception:
        return None


def imResize(im, size, fit='height'):
    # resize a float 0-1 image. 'height' reproduces the original adaCG scaling
    # (image height set to screen width / rescale), 'long' bounds the long edge instead.
    if fit == 'width':
        sz = np.divide(size, im.shape[1])
    elif fit == 'long':
        sz = np.divide(size, max(im.shape[0], im.shape[1]))
    else:
        sz = np.divide(size, im.shape[0])
    out = Image.fromarray(np.multiply(np.clip(im,0,1),255).astype(np.uint8))
    out = out.resize((max(1,int(im.shape[1]*sz)), max(1,int(im.shape[0]*sz))))
    # float32, not float64. halves the memory held per preview and speeds up LUT.apply,
    # which matters once a whole project is held in memory at once.
    return np.divide(np.array(out, dtype=np.float32), np.float32(255))


def newRecord(im, nBins=NBINS, histMode='std', thumbScale=0.05, labels=None):
    # build a single gradeDict entry for an image
    return {'set': defaultSet(),
            'reg': defaultReg(),
            'hist': imHist(im, nBins, histMode),
            'thumb': imThumb(im, thumbScale),
            'labels': list(labels) if labels else [],
            'time': datetime.now()}


class adapt():

    def __init__(self, gd, tList, nBins=NBINS, nNeighbors=16):

        # tList is the list of gradeDict keys to train on. gd is the gradeDict itself.
        tList = [k for k in tList if k in gd and isinstance(gd[k], dict) and 'hist' in gd[k]]
        if not len(tList):
            raise ValueError('adapt: no trainable records in tList')

        self.tList = list(tList)
        self.nBins = nBins
        self.regions = list(REGIONS)
        self.model = KNeighborsRegressor(n_neighbors=min(nNeighbors,len(tList)),p=1,weights='distance')
        self.hists = np.zeros((len(tList),nBins))
        self.trts = np.zeros((len(tList),len(self.regions)))
        for i,tKey in enumerate(tList):
            h = np.asarray(gd[tKey]['hist'], dtype=np.float64).reshape(-1)
            if h.size != nBins: # guard against records histogrammed at a different bit depth
                raise ValueError('adapt: %s has a %d bin histogram, expected %d' % (tKey, h.size, nBins))
            self.hists[i,:] = h
            for j,reg in enumerate(self.regions):
                self.trts[i,j] = gd[tKey]['reg'][reg][0]
        self.model.fit(self.hists, self.trts)

    def query(self,qHist):
        # accepts a single histogram or a stack, always returns a 2D (n, 4) array of pivots
        qHist = np.atleast_2d(np.asarray(qHist, dtype=np.float64))
        return self.model.predict(qHist)

    def queryDict(self, gd, key):
        # predict the reg pivots for one gradeDict record and return them as a reg dict
        pred = self.query(gd[key]['hist'])[0]
        regOut = copy.deepcopy(gd[key]['reg'])
        for j,reg in enumerate(self.regions):
            regOut[reg][0] = float(pred[j]) # prediction only touches the pivot, falloff is left alone
        return regOut

class colorEngine():

    def __init__(self, nodes=17):

        self.lutDomain = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]]) # set LUT bounds
        self.idLut = c.LUT3D.linear_table(nodes) # create LUT table
        self.yLut = np.repeat(np.mean(self.idLut,axis=3,keepdims=True), 3,axis=3) # average pixel intensity for weight map
        xyzLut = c.RGB_to_XYZ(self.idLut, c.models.RGB_COLOURSPACE_DISPLAY_P3, apply_cctf_decoding=True) # convert to CIE XYZ
        self.xyzLut = c.LUT3D(xyzLut,'LUT XYZ', self.lutDomain)
        self.labLut = c.XYZ_to_Lab(xyzLut) # convert to CIE LAB

    def apply(self, gradeDict):

        lutOut = self.idLut.copy()

        regDict = gradeDict['reg']
        setDict = gradeDict['set']

        # set region maps with current pivot and falloff points
        mapDD = np.clip(np.piecewise(self.yLut, [self.yLut < regDict['darkest'][0], self.yLut >= regDict['darkest'][0]], [1, lambda x: regDict['darkest'][1] * x + (1 - regDict['darkest'][0] * regDict['darkest'][1])]),0,1)
        mapD = np.clip(np.piecewise(self.yLut, [self.yLut < regDict['dark'][0], self.yLut >= regDict['dark'][0]], [1, lambda x: regDict['dark'][1] * x + (1 - regDict['dark'][0] * regDict['dark'][1])]),0,1)
        mapL = np.clip(np.piecewise(self.yLut, [self.yLut <= regDict['light'][0], self.yLut > regDict['light'][0]], [lambda x: regDict['light'][1] * x + (1 - regDict['light'][0] * regDict['light'][1]), 1]),0,1)
        mapLL = np.clip(np.piecewise(self.yLut, [self.yLut <= regDict['lightest'][0], self.yLut > regDict['lightest'][0]], [lambda x: regDict['lightest'][1] * x + (1 - regDict['lightest'][0] * regDict['lightest'][1]), 1]),0,1)
        maps = {'darkest': mapDD, 'dark': mapD, 'light': mapL, 'lightest': mapLL}

        # for each image region
        adjLuts = []
        blendLuts = []
        mapLuts = []
        for reg in REGIONS:
            labLut_ = np.copy(self.labLut) # copy CIELAB ID LUT to apply a*b* offsets for each region
            labLut_[:,:,:,1] = (labLut_[:,:,:,1] + setDict[reg][0])
            labLut_[:,:,:,2] = (labLut_[:,:,:,2] + setDict[reg][1])
            xyzLut_ = c.Lab_to_XYZ(labLut_) # convert to CIE XYZ
            adjLut = c.XYZ_to_RGB(xyzLut_, c.models.RGB_COLOURSPACE_DISPLAY_P3, apply_cctf_encoding=True) # convert to display space
            # weighted average of adjLut and ID LUT based on region weight map. Out of region = ID LUT, in region = adj LUT
            blendLut = c.LUT3D((maps[reg] * adjLut) + ((1 - maps[reg]) * self.idLut), 'regionLUT', self.lutDomain)
            mapLuts.append(maps[reg])
            adjLuts.append(adjLut)
            blendLuts.append(blendLut)
            lutOut = blendLut.apply(lutOut) # recursively apply region adjustments

        lutOutC = c.LUT3D(lutOut, 'fullLut', self.lutDomain) # convert to LUT object

        return lutOutC, maps


class cgEditor(tk.Frame):
    # the grading interface itself: image view, a*b* color box, tonescale gradient, region dropdown.
    # split out of adaCGapp so dBadaCG.py can embed the same controls without duplicating them.
    # the editor holds no image list and no pkl, it only ever operates on a set/reg pair.

    def __init__(self, master, ce=None, nodes=17, bounds=50, cntrlsz=200, gradWidth=240, onChange=None, **kw):

        super().__init__(master, bg='gray9', **kw)

        self.setDict = defaultSet()
        self.regDict = defaultReg()
        self.im = None # working (display resolution) image, float 0-1
        self.onChange = onChange # fired after any grade edit, so the host can write back to its gradeDict
        self.regVal = 'light'
        self.regMask = False
        self.lastLut = None # LUT built by the most recent update_image, see dBadaCG.lutFor

        # init color engine
        self.ce = ce if ce is not None else colorEngine(nodes)
        self.lutDomain = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]]) # set LUT bounds
        self.idLut = c.LUT3D.linear_table(nodes) # create LUT table
        self.redLut = 1 - self.idLut.copy()
        #self.redLut[:,:,:,1:3] = np.multiply(self.redLut[:,:,:,1:3],0) # region visualization mask - isolate red channel

        # image pane
        self.image_label = tk.Label(self,bg='gray9')
        self.image_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # controls pane
        self.control_frame = tk.Frame(self, bg='gray9')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # init color box UI
        self.bounds = bounds # color correction bounds (CIELAB a*b* radius)
        self.cntrlsz = cntrlsz # color box size (in pixels)
        aPlot = np.repeat(np.reshape(np.linspace(-self.bounds,self.bounds,self.cntrlsz),(1,self.cntrlsz,1)),self.cntrlsz,axis=0) # init a*b* channels for UI box
        bPlot = np.repeat(np.reshape(np.linspace(-self.bounds,self.bounds,self.cntrlsz),(self.cntrlsz,1,1)),self.cntrlsz,axis=1)
        Lplot = np.ones((self.cntrlsz,self.cntrlsz,1)) * 60 # set color box L channel
        cBoxLAB = np.dstack([Lplot,aPlot,bPlot]) # convert to display space
        cBoxXYZ = c.Lab_to_XYZ(cBoxLAB)
        cBoxRGB = np.clip(c.XYZ_to_RGB(cBoxXYZ, c.models.RGB_COLOURSPACE_DISPLAY_P3, apply_cctf_encoding=True),0,1)
        cBoxRGB = Image.fromarray(np.multiply(cBoxRGB,255).astype(np.uint8))
        # add grid
        gSize = int(np.floor(self.cntrlsz/2)) # grid size (subdivide color box by 2)
        cBoxRGBdraw = ImageDraw.Draw(cBoxRGB)
        for x in range(0,self.cntrlsz,gSize):
            cBoxRGBdraw.line([(x, 0), (x, self.cntrlsz)], fill='white', width=1)
        for y in range(0, self.cntrlsz, gSize):
            cBoxRGBdraw.line([(0, y), (self.cntrlsz, y)], fill='white', width=1)

        # color controls
        self.color_wheel_img = ImageTk.PhotoImage(cBoxRGB) # init color box object
        self.canvas_wheel = tk.Canvas(self.control_frame, width=self.cntrlsz, height=self.cntrlsz, highlightthickness=0)
        self.canvas_wheel.pack(pady=5)
        self.canvas_wheel.create_image(0, 0, anchor=tk.NW, image=self.color_wheel_img)
        self.canvas_wheel.bind("<Button-1>", self.on_wheel_click) # bind color box interaction to mouse actions
        self.canvas_wheel.bind("<B1-Motion>", self.on_wheel_drag)

        # region dropdown
        self.region = tk.StringVar(self.control_frame,'light') # set initial drop down value
        self.reg = tk.OptionMenu(self.control_frame,self.region,*REGIONS,command=self.setReg) # init region drop down
        self.reg.configure(bg='gray9',fg='gray99',highlightthickness=0)
        self.reg.pack(pady=5)

        # init black-to-white gradient visualization
        self.gradient_size = [int(gradWidth/10), int(gradWidth)] # set width to image width, set height to image width/10
        self.grad = (np.repeat(np.repeat(np.reshape(np.linspace(0,1,self.gradient_size[1]),(1,self.gradient_size[1],1)),self.gradient_size[0],axis=0),3,axis=2))
        self.gradPIL = Image.fromarray(np.multiply(self.grad,255).astype(np.uint8))
        self.gradTK = ImageTk.PhotoImage(self.gradPIL) # init gradient object
        self.gradient_canvas = tk.Canvas(self.control_frame, width=self.gradient_size[1], height=self.gradient_size[0], highlightthickness=0)
        self.gradient_canvas.create_image(0, 0, anchor=tk.NW, image=self.gradTK)
        self.gradient_canvas.pack(pady=5)

        self.pivLine = None # init pivot point interaction
        self.gradient_canvas.bind("<B1-Motion>", self.on_line_drag)
        self.gradient_canvas.bind("<ButtonRelease-1>", lambda e: self.stop_drag())

        self.wheel_marker = None
        self.draw_wheel_marker(self.cntrlsz/2, self.cntrlsz/2) # place color marker in center of color box

    # ---- host interface -------------------------------------------------

    def setImage(self, im, refresh=True):
        # hand the editor a new (already resized) float 0-1 image
        self.im = im
        if refresh:
            lutOut = self.update_image()
            self.update_gradient(lutOut)

    def setGrade(self, setDict, regDict, refresh=True):
        # load a set/reg pair out of a gradeDict record
        self.setDict = copy.deepcopy(setDict)
        self.regDict = copy.deepcopy(regDict)
        self.draw_wheel_marker(0,0) # coordinates are recomputed from setDict inside
        if refresh:
            lutOut = self.update_image()
            self.update_gradient(lutOut)

    def getGrade(self):
        # deep copies so the caller cannot alias the editor's live dicts
        return copy.deepcopy(self.setDict), copy.deepcopy(self.regDict)

    def gradeDict(self):
        # shape the editor state the way colorEngine.apply expects it
        return {'set': self.setDict, 'reg': self.regDict}

    def notify(self):
        # tell the host a grade edit happened
        if self.onChange is not None:
            self.onChange(self.setDict, self.regDict)

    def reset(self):
        # reset controls to initial values
        self.setDict = defaultSet()
        self.draw_wheel_marker(self.cntrlsz/2, self.cntrlsz/2) # update wheel marker
        lutOut = self.update_image() # update image
        self.update_gradient(lutOut) # update gradient
        self.notify()

    # ---- interaction ----------------------------------------------------

    def on_wheel_click(self, event):
        # click on the color box and apply offset according to click location
        self.update_color_from_wheel(event.x, event.y)

    def on_wheel_drag(self, event):
        # drag color marker and apply offset according to current mouse location in real time
        self.update_color_from_wheel(event.x, event.y)

    def update_color_from_wheel(self, x, y):
        # convert marker pixel location to a*b* offset for the current selected region
        # (pixel distance from center of box, max normalized by box size and scaled to a*b* diameter of control box)
        self.setDict[self.regVal][0] = ((x - self.cntrlsz/2) / self.cntrlsz) * self.bounds * 2
        self.setDict[self.regVal][1] = ((y - self.cntrlsz/2) / self.cntrlsz) * self.bounds * 2
        self.draw_wheel_marker(x, y) # move color marker
        lutOut = self.update_image() # update image with color edit and return LUT
        self.update_gradient(lutOut) # update gradient visualization
        self.notify()

    def draw_wheel_marker(self, x, y):
        radius = 5 # marker size
        # convert a*b* offset for the current selected region to marker pixel coordinate (inverse to update_color_from_wheel)
        x = (self.setDict[self.regVal][0] / (self.bounds * 2)) * self.cntrlsz + (self.cntrlsz/2)
        y = (self.setDict[self.regVal][1] / (self.bounds * 2)) * self.cntrlsz + (self.cntrlsz/2)
        # delete the old marker
        if self.wheel_marker:
            self.canvas_wheel.delete(self.wheel_marker)
        # draw the new marker
        self.wheel_marker = self.canvas_wheel.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline="black", width=2
        )

    def on_line_drag(self,event):
        # update region pivot by moving red dotted line on gradient visualization
        x = max(0, min(self.gradient_size[1], event.x)) # keep control in bounds of gradient visualization
        self.regDict[self.regVal][0] = x / self.grad.shape[1] # convert from pixel location to relative intensity
        self.regMask = True # while dragging, turn on region mask visualization
        lutOut = self.update_image() # update image with new region control
        self.update_gradient(lutOut) # update gradient with new LUT

    def stop_drag(self):
        # when dragging stops, apply the final updates to region maps, image, LUT, gradient visualization
        self.regMask = False # turn off region mask visualization
        lutOut = self.update_image()
        self.update_gradient(lutOut)
        self.notify()

    # record regVal for shadows, highlights
    def setReg(self,value):
        # select region from dropdown menu
        self.regVal = value
        self.region.set(value)
        self.draw_wheel_marker(0,0)
        lutOut = self.update_image() # query current LUT
        self.update_gradient(lutOut) # update gradient visualization for current region

    # ---- render ---------------------------------------------------------

    def update_gradient(self,lutOut):
        # update gradient visualization with new edits
        gradOut = np.clip(lutOut.apply(self.grad),0,1) # clip LUT to 0-1 range
        self.gradPIL = Image.fromarray(np.multiply(gradOut,255).astype(np.uint8))
        self.gradTK = ImageTk.PhotoImage(self.gradPIL) # send current gradient to visualization box
        self.gradient_canvas.delete("all")
        self.gradient_canvas.create_image(0, 0, anchor=tk.NW, image=self.gradTK)
        pivPix = int(self.grad.shape[1] * self.regDict[self.regVal][0]) # convert pivot point (relative intensity) to gradient pixel location
        fall = self.regDict[self.regVal][1]
        fall = fall if fall != 0 else 1e-6 # a zero falloff is an infinitely wide region, keep the line drawable
        slopePix = pivPix - int((1/fall) * self.grad.shape[1]) # set point where falloff intersects x-axis
        self.gradient_canvas.create_line(pivPix, 0, pivPix, self.grad.shape[0], fill="red", width=2, tags="pivLine", dash = (4,2)) # create pivot line (vertical)
        self.gradient_canvas.create_line(slopePix,self.grad.shape[0],pivPix,3, fill="red", width=2, tags="pivLine", dash = (4,2))  # create falloff line
        # create horizontal region bound line from black to pivot point for black or dark, or from pivot point to white for midtones and highlights
        if self.regVal == 'darkest':
            self.gradient_canvas.create_line(0,3,pivPix,3, fill="red", width=2, tags="pivLine", dash = (4,2)) # draw line 3 pixels from top of gradient for visibility
        elif self.regVal == 'dark':
            self.gradient_canvas.create_line(0,3,pivPix,3, fill="red", width=2, tags="pivLine", dash = (4,2)) # draw line 3 pixels from top of gradient for visibility
        else:
            self.gradient_canvas.create_line(pivPix,3,self.grad.shape[1],3, fill="red", width=2, tags="pivLine", dash = (4,2))

    def update_image(self):

        lutOutC, maps = self.ce.apply(self.gradeDict())
        self.lastLut = lutOutC # held so callers rendering this same grade need not rebuild it

        if self.im is None: # nothing loaded yet, still return the LUT so the gradient can draw
            self.image_label.config(image='') # do not leave the previous image up
            return lutOutC

        im = lutOutC.apply(self.im) # apply LUT to image

        if self.regMask: # if dragging pivot point, apply region mask visualization LUT to output image
            maskLut = c.LUT3D((maps[self.regVal] * self.redLut) + ((1 - maps[self.regVal]) * self.idLut), 'maskLUT', self.lutDomain)
            im = maskLut.apply(im)

        imOut = np.clip(im,0,1)
        pilImage = Image.fromarray(np.multiply(imOut,255).astype(np.uint8))
        self.imsTK = ImageTk.PhotoImage(pilImage)
        self.image_label.config(image=self.imsTK) # send image to UI

        return lutOutC


class adaCGapp(tk.Tk):

    def __init__(self, imPath, ftype):

        super().__init__()
        self.title("adaCG: Adaptive Color Grading")

        # query monitor info
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.width = screen_width
        self.height = screen_height

        # initialize dictionaries
        self.setDict = defaultSet() # init a*b* offsets
        self.regDict = defaultReg() # [pivot, falloff] for each region

        # init grade file
        self.resF = os.path.join(imPath, 'grade.pkl')
        if not os.path.exists(self.resF):
            self.gradeDict = {}
            with open(self.resF, "wb") as file:
                pickle.dump(self.gradeDict, file)
        else:
            with open(self.resF, "rb") as file:
                self.gradeDict = pickle.load(file)

        # load images
        rescale = 8 # image scaled to (screen width / rescale)
        self.nBins = NBINS
        self.imPath = imPath
        self.imList = [ f for f in os.listdir(imPath) if f.endswith(ftype) ] # load all images in directory of given type
        self.nIms = len(self.imList)
        self.ims = [0] * self.nIms

        self.imSelect = 0
        print('loading images ..')
        self.gradeDict.setdefault('root', imPath)
        for i in range(self.nIms):
            fullIm = c.read_image(os.path.join(imPath,self.imList[i])) # loads image as float32 in 0-1 range
            self.ims[i] = imResize(fullIm, self.width/rescale)
            # fill in any image that has no record yet, so a directory that gained files
            # after the pkl was written still grades. setdefault never clobbers an existing grade.
            if self.imList[i] not in self.gradeDict:
                self.gradeDict[self.imList[i]] = newRecord(fullIm, self.nBins)

        # init color engine
        nodes = 17 # set number of nodes
        self.ce = colorEngine(nodes)

        # the whole grading interface now lives in cgEditor
        self.editor = cgEditor(self, ce=self.ce, nodes=nodes, gradWidth=int(self.width/rescale))
        self.editor.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # image controls
        self.control_frame = tk.Frame(self, bg='gray9') # init app window & buttons
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.fIm = tk.Button(self.control_frame, text="Next Image",command=self.forwardIm,bg='gray9',fg='gray99')
        self.fIm.pack(side=tk.RIGHT)
        self.bIm = tk.Button(self.control_frame, text="Prev Image",command=self.backIm,bg='gray9',fg='gray99')
        self.bIm.pack(side=tk.LEFT)
        self.bIm = tk.Button(self.control_frame, text="Reset",command=self.reset,bg='gray9',fg='gray99')
        self.bIm.pack(pady=5)
        self.sIm = tk.Button(self.control_frame, text="Save Image",command=self.saveIm,bg='gray9',fg='gray99')
        self.sIm.pack(pady=5)
        self.saveLut = tk.Button(self.control_frame, text="Save Lut",command=self.saveLUT,bg='gray9',fg='gray99')
        self.saveLut.pack(pady=5)
        self.sGrade = tk.Button(self.control_frame, text="Save Grade",command=self.saveGrade,bg='gray9',fg='gray99')
        self.sGrade.pack(pady=5)
        self.lGrade = tk.Button(self.control_frame, text="Load Grade",command=self.loadGrade,bg='gray9',fg='gray99')
        self.lGrade.pack(pady=5)

        self.title("rawGrader: "+self.imList[self.imSelect])
        self.editor.setImage(self.ims[self.imSelect]) # pass image through initial pipeline

    def forwardIm(self):
        # when button is clicked, move to next image in list
        if self.imSelect < self.nIms-1:
            self.imSelect += 1
        else:
            self.imSelect = 0 # if at end of list, go back to beginning
        self.editor.setImage(self.ims[self.imSelect]) # pass new image through pipeline
        self.title("rawGrader: "+self.imList[self.imSelect]) # update label

    def backIm(self):
        # when button is clicked, move to the previous image in list
        if self.imSelect > 0:
            self.imSelect -= 1
        else:
            self.imSelect = self.nIms-1 # if at beginning of list, go back to the end
        self.editor.setImage(self.ims[self.imSelect]) # pass new image through pipeline
        self.title("rawGrader: "+self.imList[self.imSelect])

    def reset(self):
        self.editor.reset()

    def saveIm(self):
        # save full sized edited image
        lutOut = self.editor.update_image() # return current LUT object
        fullIm = c.read_image(os.path.join(self.imPath,self.imList[self.imSelect])) # loads image and as float32 in 0-1 range
        imOut = np.clip(lutOut.apply(fullIm),0,1) # apply LUT
        c.write_image(imOut,'rawGradeOut.jpg',bit_depth='uint8') # write image to 8-bit jpg

    def saveLUT(self):
        # save lut as .cube file
        lutOut = self.editor.update_image() # return current LUT object
        c.write_LUT(lutOut, 'lut.cube') # write to .cube file

    def saveGrade(self):
        # update grade dict
        setDict, regDict = self.editor.getGrade()
        self.gradeDict[self.imList[self.imSelect]]['set'] = setDict
        self.gradeDict[self.imList[self.imSelect]]['reg'] = regDict
        self.gradeDict[self.imList[self.imSelect]]['time'] = datetime.now()
        # save settings to pkl
        with open(self.resF, "wb") as file:
            pickle.dump(self.gradeDict, file)

    def loadGrade(self):
        # if the file has been graded before, load previous grade
        rec = self.gradeDict[self.imList[self.imSelect]]
        self.editor.setGrade(rec['set'], rec['reg'])

if __name__ == "__main__":

    import os
    #impath = os.path.normpath("c:/Users/t.canham/Documents/photo-finishing/images/2HDRVD/choiceFramesP3")
    #impath = os.path.normpath("c:/Users/t.canham/Documents/photo-finishing/images/2HDRVD")
    #impath = os.path.normpath("c:/Users/t.canham/Documents/photo-finishing/images/hdr4eu/P3")
    impath = os.path.normpath("c:/Users/t.canham/Documents/photo-finishing/images/2HDRVD/expPrelim/")
    ftype = ".tif"
    #resF = os.path.normpath("c:/Users/t.canham/Documents/photo-finishing/results/resExpRawGrader.xlsx")
    app = adaCGapp(impath,ftype)
    app.mainloop()
