
import os, sys, glob, runpy, importlib
import numpy as np

NABAT_DIR = '/home/jgersey/SageBat/nabat-ml'
BOUT_GLOB = '/home/jgersey/SageBat/bouts_ab/*_notched.wav'
NPZ       = '/home/jgersey/SageBat/joint_per_bout.npz'
OUT_DIR   = os.path.join(NABAT_DIR, "classify_joint")

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, NABAT_DIR)
os.chdir(NABAT_DIR)

CAPTURED = []
PredictionCls = None
for modname in ("prediction.prediction", "prediction"):
    try:
        m = importlib.import_module(modname)
    except Exception:
        continue
    if hasattr(m, "Prediction"):
        PredictionCls = m.Prediction; break
if PredictionCls is None:
    sys.exit("could not locate Prediction class")

_orig = PredictionCls.__init__
def _patched(self, *a, **kw):
    _orig(self, *a, **kw)
    model = getattr(self, "MODEL", None)
    if model is None: return
    orig_predict = model.predict
    def wrapper(*aa, **kk):
        out = orig_predict(*aa, **kk)
        try:
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[1] > 5:
                CAPTURED.append(arr.copy())
        except Exception:
            pass
        return out
    model.predict = wrapper
PredictionCls.__init__ = _patched

out = {}
for wav in sorted(glob.glob(BOUT_GLOB)):
    CAPTURED.clear()
    sys.argv = ["nabat_ml_cli.py", "-p", OUT_DIR, "-f", wav]
    key = os.path.splitext(os.path.basename(wav))[0]
    print("---", key)
    try:
        runpy.run_path(os.path.join(NABAT_DIR, "nabat_ml_cli.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print("  error:", e)
    if CAPTURED:
        best = max(CAPTURED, key=lambda a: a.shape[0])
        out[key] = best
        print("  captured", best.shape)

if not out:
    sys.exit("NO_VECTORS")
np.savez(NPZ, **out)
print("saved", list(out), "->", NPZ)
