# -*- coding: utf-8 -*-
"""Bam duong tam bang CNN chay tren TensorRT. Thay the DUNG BUOC TAO MASK.

Doi lai cua `lane.py` (CV co dien) chi la nguon mask:

    lane.py      anh -> nguong HSV -> mask VET SON  -\
                                                      >-- warp -> band -> fit -> cte -> PID
    lane_cnn.py  anh -> TensorRT   -> mask DUONG TAM -/

Tra ve dung `LaneResult` cua lane.py nen `pipeline.py`, PID, FSM, driver khong
phai sua mot dong nao.

VI SAO HAU XU LY O DAY NGAN HON lane.py NHIEU:
lane.py phai gan loc vien lane, chon cum theo anchor, chong nhay - vi mask mau
cua no lan lon ca vien lane cung mau. Mask CNN chi co DUNG MOT dai lien tuc,
khong vien khong dom khong khoang trong, nen chi con: chia band -> lay cum manh
nhat -> fit. Do la loi cua viec dat nhan la duong lien tuc thay vi cac vet son.

CHUOI TIEN/HAU XU LY PHAI GIONG HET LUC TRAIN.
Cac hang so hinh hoc duoi day sao chep tu `training/lane_post.py`. Sua mot ben
ma khong sua ben kia thi model KHONG BAO LOI GI - no chi te di. Doi luon la doi
CA HAI, va export lai ONNX.

Co y KHONG doc `lane.warp_src` tu config: khoa do dung de tune duong CV. Ai do
chinh no cho CV se lam lech am tham hinh hoc cua CNN. Muon doi rieng cho CNN thi
dung khoa `lane.cnn.*`.

Python 3.6 (JetPack 4.5.1, TensorRT 7.1.3 - DA XAC NHAN TREN XE).
Can: tensorrt (co san theo JetPack), pycuda (cai them).
Cai pycuda PHAI xuat bien CUDA truoc, khong thi loi 'cuda.h: No such file':
    export PATH=/usr/local/cuda/bin:$PATH
    export CPATH=/usr/local/cuda/include:$CPATH
    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
    pip3 install --user --no-cache-dir pycuda      # KHONG dung sudo
"""

from __future__ import print_function

import os
import time

import cv2
import numpy as np

from .lane import LaneResult

# --- Sao chep tu training/lane_post.py. DOI THI DOI CA HAI. ----------------
PROC_W, PROC_H = 320, 240
ROI_TOP = 0.55
ROI_Y0 = int(PROC_H * ROI_TOP)
IN_W, IN_H = 256, 128            # dau vao model
SEG_W, SEG_H = 128, 64           # dau ra seg (stride 2)
WARP_SRC = [[0.10, 1.00], [0.90, 1.00], [0.62, 0.58], [0.38, 0.58]]
WARP_DST_MARGIN = 0.25
N_BANDS = 10
BAND_MIN_PIXELS = 10
MIN_BANDS = 2
MAX_RUN_FRAC = 0.60
LOOKAHEAD = 0.6
# ---------------------------------------------------------------------------


def _build_warp(w, h, src_pts, margin):
    src = np.float32([[p[0] * w, p[1] * h] for p in src_pts])
    dst = np.float32([[margin * w, h], [(1.0 - margin) * w, h],
                      [(1.0 - margin) * w, 0.0], [margin * w, 0.0]])
    return cv2.getPerspectiveTransform(src, dst)


class TensorRTEngine(object):
    """Boc engine TensorRT. Ho tro ca API binding (TRT 8.0) va tensor (TRT >=8.5).

    Vi sao ho tro ca hai: xe chay TRT 7.1.3 dung API binding, nhung may dev hoac
    ban JetPack khac co the la 8.5+ noi API do da bo. Doan if/else nay re hon
    nhieu so voi mot buoi debug vi doi image.
    """

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit      # noqa: F401  tao CUDA context
        except ImportError as exc:
            raise ImportError(
                '%s.\n'
                'lane.mode=cnn CHI chay duoc tren Jetson. tensorrt co san theo\n'
                'JetPack; pycuda phai cai them va PHAI xuat bien CUDA truoc,\n'
                'neu khong se loi "cuda.h: No such file or directory":\n'
                '  export PATH=/usr/local/cuda/bin:$PATH\n'
                '  export CPATH=/usr/local/cuda/include:$CPATH\n'
                '  export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH\n'
                '  pip3 install --user --no-cache-dir pycuda   (KHONG dung sudo:\n'
                '  sudo xoa sach cac bien vua xuat)\n'
                'Tren laptop: dat lane.cnn.backend=onnx de chay bang ONNXRuntime,\n'
                'hoac lane.cnn.engine rong de chi dung hau xu ly.'
                % exc)
        self._cuda = cuda

        if not os.path.isfile(engine_path):
            raise IOError(
                'Khong thay engine: %s\n'
                'Build TREN CHINH XE:\n'
                '  /usr/src/tensorrt/bin/trtexec --onnx=lane_tiny.onnx '
                '--fp16 --workspace=256 --saveEngine=%s'
                % (engine_path, engine_path))

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as fh:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(
                'TensorRT khong doc duoc %s. Gan nhu chac chan engine nay build '
                'tren may/phien ban khac - engine KHONG chuyen may duoc.'
                % engine_path)
        self.context = self.engine.create_execution_context()
        self._new_api = hasattr(self.engine, 'num_io_tensors')

        # THU TU BINDING CUA ENGINE KHONG PHAI THU TU KHAI BAO TRONG ONNX.
        # Do binding nao la vao/ra bang API cua engine, va tra ket qua theo TEN.
        # Ban dau file nay gia dinh outs[0] = seg, outs[1] = reg -> tren xe
        # TensorRT tra reg truoc va no gay "cannot reshape array of size 3 into
        # shape (64,128)". Loi chi lo ra khi co engine that, khong cach nao bat
        # duoc tren laptop.
        self.host, self.dev, self.names, self.shapes = [], [], [], []
        self.in_idx, self.out_idx = [], []
        if self._new_api:
            for i in range(self.engine.num_io_tensors):
                nm = self.engine.get_tensor_name(i)
                is_in = (self.engine.get_tensor_mode(nm) ==
                         trt.TensorIOMode.INPUT)
                self._alloc(nm, tuple(self.engine.get_tensor_shape(nm)),
                            trt.nptype(self.engine.get_tensor_dtype(nm)), is_in)
        else:
            for i in range(self.engine.num_bindings):
                self._alloc(self.engine.get_binding_name(i),
                            tuple(self.engine.get_binding_shape(i)),
                            trt.nptype(self.engine.get_binding_dtype(i)),
                            self.engine.binding_is_input(i))
        if not self.in_idx or not self.out_idx:
            raise RuntimeError('Engine khong co du input/output: %s' % self.names)
        self.stream = cuda.Stream()
        self.bindings = [int(d) for d in self.dev]

    def _alloc(self, name, shape, dtype, is_input):
        idx = len(self.host)
        host = self._cuda.pagelocked_empty(int(np.prod(shape)), dtype)
        self.host.append(host)
        self.dev.append(self._cuda.mem_alloc(host.nbytes))
        self.names.append(name)
        self.shapes.append(shape)
        (self.in_idx if is_input else self.out_idx).append(idx)

    def infer(self, x):
        """x: float32 (1,3,H,W) lien tuc -> dict {ten: mang}."""
        cuda = self._cuda
        i0 = self.in_idx[0]
        np.copyto(self.host[i0], x.ravel())
        cuda.memcpy_htod_async(self.dev[i0], self.host[i0], self.stream)
        if self._new_api:
            for nm, d in zip(self.names, self.dev):
                self.context.set_tensor_address(nm, int(d))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings,
                                          stream_handle=self.stream.handle)
        for i in self.out_idx:
            cuda.memcpy_dtoh_async(self.host[i], self.dev[i], self.stream)
        self.stream.synchronize()
        return dict((self.names[i], self.host[i].reshape(self.shapes[i]))
                    for i in self.out_idx)


class OnnxEngine(object):
    """Chay bang ONNXRuntime thay TensorRT. CHAM HON NHIEU - khong dung de thi.

    Ly do co mat: cho phep chay TOAN BO duong deploy (preprocess -> model ->
    hau xu ly -> cte) tren laptop, khong can Jetson. Nho no, phan duy nhat chua
    duoc kiem chung khi len xe chi con dung mot thu: TensorRT. Ngoai ra no la
    luoi an toan neu trtexec khong build duoc vao phut chot.
    """

    def __init__(self, onnx_path):
        import onnxruntime as ort
        if not os.path.isfile(onnx_path):
            raise IOError('Khong thay file ONNX: %s' % onnx_path)
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            onnx_path, opt, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def infer(self, x):
        outs = self.sess.run(None, {self.input_name: x})
        return dict(zip(self.out_names, outs))


class CnnLaneDetector(object):
    """Cung giao dien voi LaneDetector: .process(frame_bgr) -> LaneResult."""

    def __init__(self, cfg):
        c = cfg.get
        engine = c('lane.cnn.engine', 'models/lane_tiny.engine')
        # 'trt' = TensorRT (tren xe). 'onnx' = ONNXRuntime (kiem chung tren PC).
        self.backend = str(c('lane.cnn.backend', 'trt'))
        self.roi_top = float(c('lane.cnn.roi_top', ROI_TOP))
        self.engine_path = engine
        self.update_config(cfg)
        self.M = _build_warp(PROC_W, PROC_H,
                             c('lane.cnn.warp_src', WARP_SRC),
                             float(c('lane.cnn.warp_dst_margin', WARP_DST_MARGIN)))
        self.roi_y0 = int(PROC_H * self.roi_top)
        # `engine: ''` -> chi dung duoc hau xu ly, khong suy dien. Duong nay ton
        # tai de doi chieu hau xu ly cua XE voi training/lane_post.py NGAY TREN
        # LAPTOP - neu hai ben lech nhau thi moi con so do o PC deu vo nghia.
        if not engine:
            self.engine = None
        elif self.backend == 'onnx':
            self.engine = OnnxEngine(engine)
        else:
            self.engine = TensorRTEngine(engine)

        self.last_infer_ms = 0.0
        self.last_total_ms = 0.0
        self.last_mask = None      # mask ROI 256x128 do model xuat ra
        self.last_bev = None       # cung mask do sau khi warp sang BEV
        self.n_disagree = 0
        self._reset_state()
        if self.engine is not None:
            self._warmup()

    def update_config(self, cfg):
        """Doc lai cac tham so RE tu config. KHONG dung toi engine TensorRT.

        Ton tai vi giao dien tune goi `rebuild()` moi lan keo slider. Neu de no
        dung lai ca detector thi moi lan keo se nap lai engine: treo giao dien
        vai giay, va cap phat lai bo nho GPU ma khong giai phong cai cu.
        Nhung tham so duoi day deu chi la so - doi tuc thi, khong ton gi.
        """
        c = cfg.get
        self.alpha = float(c('lane.smooth_alpha', 0.6))
        self.min_bands = int(c('lane.cnn.min_bands', MIN_BANDS))
        self.band_min_pixels = int(c('lane.cnn.band_min_pixels', BAND_MIN_PIXELS))
        # Diem ngam xa: truoc day la hang so nen slider khong co tac dung gi -
        # nguoi tune keo mai ma khong hieu sao xe khong doi.
        self.lookahead = float(c('lane.cnn.lookahead', LOOKAHEAD))
        # Nguong bat dong y giua seg head va reg head. Vuot nguong = hai duong
        # tinh doc lap trong model cho ket qua khac nhau -> khong tin duoc nua,
        # bao found=False de FSM ha ga / roi ve CV.
        self.reg_disagree = float(c('lane.cnn.reg_disagree', 0.35))
        self.check_reg = bool(c('lane.cnn.check_reg', True))

    def _reset_state(self):
        self._cte = 0.0
        self._look = 0.0
        self._curv = 0.0
        self._init = False

    def reset(self):
        self._reset_state()

    def _warmup(self, n=3):
        """Lan suy dien dau tien cua TensorRT ton hang tram ms. Dot no o day,
        khong phai o frame dau tien cua luot thi."""
        x = np.zeros((1, 3, IN_H, IN_W), np.float32)
        for _ in range(n):
            self.engine.infer(x)

    # ------------------------------------------------------------ tien xu ly
    def preprocess(self, frame_bgr):
        """PHAI GIONG HET luc train:
        crop hang [roi_top*H, H] tren anh GOC -> resize 256x128 INTER_AREA
        -> BGR (KHONG doi RGB) -> float32 thang 0..255 NCHW.
        Chuan hoa nam TRONG graph ONNX nen o day khong chia 255, khong tru mean.
        """
        h = frame_bgr.shape[0]
        roi = frame_bgr[int(h * self.roi_top):, :]
        small = cv2.resize(roi, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = small.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return np.ascontiguousarray(x)

    # ------------------------------------------------------------- hau xu ly
    def _band_centre(self, band):
        cols = np.count_nonzero(band, axis=0)
        if int(cols.sum()) < self.band_min_pixels:
            return None, 0
        hot = (cols > 0).view(np.int8)
        edges = np.diff(np.concatenate(([0], hot, [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        max_run = MAX_RUN_FRAC * PROC_W
        best_x, best_n = None, 0
        for s, e in zip(starts, ends):
            if (e - s) > max_run:
                continue
            w = cols[s:e].astype(np.float32)
            n = float(w.sum())
            if n <= best_n:
                continue
            xs = np.arange(s, e, dtype=np.float32)
            best_x, best_n = float((w * xs).sum() / n), n
        if best_x is None or best_n < self.band_min_pixels:
            return None, 0
        return best_x, int(best_n)

    def mask_to_fit(self, mask):
        """mask 256x128 (0/255) -> (coeff, t_max, n_bands, n_pixels)."""
        roi_h = PROC_H - self.roi_y0
        m = cv2.resize(mask, (PROC_W, roi_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((PROC_H, PROC_W), np.uint8)
        canvas[self.roi_y0:, :] = m
        bev = cv2.warpPerspective(canvas, self.M, (PROC_W, PROC_H))
        # Giu lai de giao dien tune ve duong fit chong len - `lane.py` cung tra
        # anh BEV trong `LaneResult.debug`, giu dung quy uoc do thi panel BEV
        # cua tuning_ui khong phai biet minh dang xem CV hay CNN.
        self.last_bev = bev

        band_h = max(1, PROC_H // N_BANDS)
        pts, npix = [], 0
        for b in range(N_BANDS):
            y1 = PROC_H - b * band_h
            y0 = max(0, y1 - band_h)
            if y1 <= 0:
                break
            cx, n = self._band_centre(bev[y0:y1, :])
            if cx is None:
                continue
            pts.append(((b + 0.5) / float(N_BANDS), cx))
            npix += n
        if len(pts) < self.min_bands:
            return None, 0.0, len(pts), npix
        ts = np.array([p[0] for p in pts], np.float64)
        xs = np.array([p[1] for p in pts], np.float64)
        co = np.polyfit(ts, xs, 2 if len(pts) >= 3 else 1)
        if len(co) == 2:
            co = np.array([0.0, co[0], co[1]])
        return co, float(ts.max()), len(pts), npix

    @staticmethod
    def _pick_outputs(outs):
        """Lay seg/reg tu dict output. Uu tien TEN, du phong theo KICH THUOC.

        Du phong ton tai vi mot so ban TensorRT doi ten binding khi build engine.
        Hai dau ra co kich thuoc chenh nhau rat xa (8192 vs 3) nen phan biet bang
        kich thuoc la an toan tuyet doi o bai nay.
        """
        seg, reg = outs.get('seg'), outs.get('reg')
        if seg is None or reg is None:
            for v in outs.values():
                if v.size == SEG_H * SEG_W:
                    seg = v
                elif v.size == 3:
                    reg = v
        if seg is None:
            raise RuntimeError(
                'Khong tim thay dau ra seg trong engine. Cac dau ra: %s'
                % dict((k, v.shape) for k, v in outs.items()))
        return seg, (None if reg is None else reg.ravel())

    # ------------------------------------------------------------------ chay
    def process(self, frame_bgr):
        if self.engine is None:
            raise RuntimeError(
                'lane.cnn.engine dang de rong - chi dung duoc hau xu ly. '
                'Dat duong dan toi file .engine da build tren xe.')
        t0 = time.time()
        x = self.preprocess(frame_bgr)

        t1 = time.time()
        outs = self.engine.infer(x)
        self.last_infer_ms = (time.time() - t1) * 1000.0

        seg, reg = self._pick_outputs(outs)
        seg = seg.reshape(SEG_H, SEG_W)

        # Nguong 0 tren logits == sigmoid > 0.5. Model co y khong co Sigmoid.
        mask = (seg > 0).astype(np.uint8) * 255
        # Hai buoc phong to nay lap Y HET duong danh gia luc train. Gop lam mot
        # buoc se cho so khac di - it thoi, nhung du de moi con so do o PC khong
        # con dung tren xe.
        mask = cv2.resize(mask, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        self.last_mask = mask

        co, t_max, n_bands, npix = self.mask_to_fit(mask)
        if co is None:
            self.last_total_ms = (time.time() - t0) * 1000.0
            return LaneResult(False, self._cte, self._curv, npix, self.last_bev,
                              cte_lookahead=self._look, n_bands=n_bands)

        half = PROC_W / 2.0
        t_look = min(self.lookahead, t_max)
        raw_cte = float(np.clip((co[2] - half) / half, -1.0, 1.0))
        raw_look = float(np.clip(
            (co[0] * t_look * t_look + co[1] * t_look + co[2] - half) / half,
            -1.0, 1.0))
        raw_curv = float(np.clip(co[0] * t_max * t_max / half, -1.0, 1.0))

        # Kiem tra cheo hai head. Chung duoc train tren cung nhan nhung di qua
        # hai duong khac nhau trong mang, nen lech nhieu = model dang doan bua.
        if self.check_reg and reg is not None and len(reg) >= 1:
            if abs(float(reg[0]) - raw_cte) > self.reg_disagree:
                self.n_disagree += 1
                self.last_total_ms = (time.time() - t0) * 1000.0
                return LaneResult(False, self._cte, self._curv, npix,
                                  self.last_bev, cte_lookahead=self._look,
                                  n_bands=n_bands)

        if not self._init:
            self._cte, self._look, self._curv = raw_cte, raw_look, raw_curv
            self._init = True
        else:
            a = self.alpha
            self._cte = a * raw_cte + (1.0 - a) * self._cte
            self._look = a * raw_look + (1.0 - a) * self._look
            self._curv = a * raw_curv + (1.0 - a) * self._curv

        self.last_total_ms = (time.time() - t0) * 1000.0
        return LaneResult(True, self._cte, self._curv, npix, self.last_bev,
                          cte_lookahead=self._look, n_bands=n_bands, fit=co)

    def debug_process(self, frame_bgr):
        res = self.process(frame_bgr)
        return {
            'small': cv2.resize(frame_bgr, (PROC_W, PROC_H)),
            'binary': self.last_mask,
            'masked': self.last_mask,
            'warped': self.last_bev,
            'roi_y': (self.roi_y0, PROC_H),
            'result': res,
            'infer_ms': self.last_infer_ms,
            'total_ms': self.last_total_ms,
        }
