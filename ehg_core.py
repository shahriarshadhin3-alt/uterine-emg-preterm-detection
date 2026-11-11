"""Signal processing + feature extraction for uterine EMG / EHG recordings."""
import numpy as np
from scipy import signal, stats

FEATURE_NAMES = [
    "mean_abs", "rms", "variance", "skewness", "kurtosis", "zero_cross_rate",
    "peak_to_peak", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "sample_entropy", "peak_freq", "median_freq", "mean_freq", "power_low",
    "power_high", "power_ratio", "spectral_entropy", "n_bursts",
    "burst_rate_per_min", "mean_burst_duration_s",
]


def bandpass(x, fs, lo=0.3, hi=3.0, order=4):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def hjorth(x):
    dx, ddx = np.diff(x), np.diff(np.diff(x))
    v0, v1, v2 = np.var(x), np.var(dx), np.var(ddx)
    mob = np.sqrt(v1 / v0) if v0 > 0 else 0.0
    comp = (np.sqrt(v2 / v1) / mob) if (v1 > 0 and mob > 0) else 0.0
    return v0, mob, comp


def sample_entropy(x, m=2, r=0.2, max_n=1000):
    x = np.asarray(x, float)
    step = max(1, len(x) // max_n)
    x = x[::step][:max_n]
    x = (x - x.mean()) / (x.std() + 1e-12)
    n = len(x)

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(emb)) / 2.0

    b, a = count(m), count(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


def find_bursts(xf, fs, min_dur_s=5.0):
    """Contraction-like bursts = smoothed envelope above mean + 0.5 SD."""
    env = np.abs(signal.hilbert(xf))
    win = max(1, int(5 * fs))
    env_s = np.convolve(env, np.ones(win) / win, mode="same")
    thr = env_s.mean() + 0.5 * env_s.std()
    edges = np.diff((env_s > thr).astype(int), prepend=0, append=0)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    bursts = [(s, e) for s, e in zip(starts, ends) if (e - s) / fs >= min_dur_s]
    return bursts, env_s, thr


def extract_features(x, fs):
    x = np.asarray(x, float)
    xf = bandpass(x, fs)
    f = {}
    f["mean_abs"] = float(np.mean(np.abs(xf)))
    f["rms"] = float(np.sqrt(np.mean(xf ** 2)))
    f["variance"] = float(np.var(xf))
    f["skewness"] = float(stats.skew(xf))
    f["kurtosis"] = float(stats.kurtosis(xf))
    f["zero_cross_rate"] = float(np.mean(np.diff(np.signbit(xf).astype(int)) != 0) * fs)
    f["peak_to_peak"] = float(np.ptp(xf))
    f["hjorth_activity"], f["hjorth_mobility"], f["hjorth_complexity"] = map(float, hjorth(xf))
    f["sample_entropy"] = sample_entropy(xf)

    freqs, psd = signal.welch(xf, fs, nperseg=min(len(xf), int(fs * 60)))
    df = freqs[1] - freqs[0]
    band = (freqs >= 0.3) & (freqs <= 3.0)
    fb, pb = freqs[band], psd[band]
    cum = np.cumsum(pb)
    f["peak_freq"] = float(fb[np.argmax(pb)])
    f["median_freq"] = float(fb[min(np.searchsorted(cum, cum[-1] / 2), len(fb) - 1)])
    f["mean_freq"] = float(np.sum(fb * pb) / np.sum(pb))
    low = (freqs >= 0.3) & (freqs < 1.0)
    high = (freqs >= 1.0) & (freqs <= 3.0)
    f["power_low"] = float(psd[low].sum() * df)
    f["power_high"] = float(psd[high].sum() * df)
    f["power_ratio"] = f["power_high"] / f["power_low"] if f["power_low"] > 0 else float("nan")
    p = pb / pb.sum()
    f["spectral_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p)))

    bursts, _, _ = find_bursts(xf, fs)
    dur_min = len(x) / fs / 60
    f["n_bursts"] = float(len(bursts))
    f["burst_rate_per_min"] = len(bursts) / dur_min if dur_min > 0 else 0.0
    f["mean_burst_duration_s"] = float(np.mean([(e - s) / fs for s, e in bursts])) if bursts else 0.0
    return f


def demo_signal(fs=20.0, minutes=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, minutes * 60, 1 / fs)
    out = {}
    for i in range(3):
        carrier = np.sin(2 * np.pi * (0.6 + 0.1 * i) * t)
        envelope = 0.2 + np.clip(np.sin(2 * np.pi * t / 120 + i), 0, None) ** 2
        out[f"Channel_{i + 1}"] = envelope * carrier + 0.3 * rng.standard_normal(len(t))
    return out
