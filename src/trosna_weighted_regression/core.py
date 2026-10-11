"""Взвешенная по времени линейная регрессия с защитой от засыпания.

Базовая модель (Ферубко, Казаков, Вестник РГРТУ, 2026, № 95): прямая y = a + b·t строится по
`window` последним точкам, вес точки обратно пропорционален её расстоянию до момента прогноза.
После долгой полки старые точки окна тянут наклон назад: на ступеньку модель отвечает выбросом
прогноза выше новой полки, на разворот тренда — запаздыванием на несколько шагов.

Защита: каждая новая точка сравнивается с прогнозом на неё. Если ошибка больше `threshold`
стандартных отклонений ошибки прогноза (шум точки с поправкой на шум самого прогноза), режим
сменился — точки до разладки выбрасываются, и модель строится заново по точкам после неё. Если
следующая точка вернулась к старому режиму, это был кратковременный всплеск: старое окно
восстанавливается, а всплеск отбрасывается.

Дополнительно (по умолчанию выключены или не влияют на прогноз):

* адаптивная длина окна по правилу пересечения доверительных интервалов (ICI, Гольденшлюгер —
  Немировский, Катковник): самое длинное окно, прогноз которого ещё согласуется с прогнозами всех
  более коротких; на ровном участке окно растёт и прогноз тише, на изломе — сразу укорачивается;
* накопительный детектор CUSUM (Пейдж) на тех же нормированных ошибках прогноза — ловит
  устойчивые сдвиги меньше порога разовой разладки, на которых длинное окно засыпало бы;
* интервал прогноза: квантиль нормированных ошибок прогноза с онлайн-подстройкой уровня
  (адаптивный конформный вывод, Гиббс — Кандес) — верхняя граница для резервирования памяти.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

KERNELS = ("metric", "exponential", "uniform")

# Переход от медианы абсолютных остатков к σ нормального шума.
_MAD_TO_SIGMA = 1.4826
# Сколько остатков нужно, чтобы медиана давала хоть сколько-нибудь надёжную оценку шума.
_MIN_RESIDUALS = 5
# Кандидаты адаптивного окна: 3, 5, 8, 12, … — геометрическая сетка, как принято в ICI.
_ICI_MIN_WINDOW = 3
_ICI_RATIO = 1.5


def kernel_weights(t, t_query, kernel="metric", power=1.0, bandwidth=0.5):
    """Нормированные веса точек `t` для прогноза в момент `t_query`.

    * ``"metric"`` — w_i ∝ |t_query − t_i|^(−power);
    * ``"exponential"`` — w_i ∝ exp(−(t_query − t_i)² / (2·bandwidth²));
    * ``"uniform"`` — равные веса (обычный МНК).
    """
    t = np.asarray(t, dtype=float)
    d = np.abs(t_query - t)
    if kernel == "uniform":
        w = np.ones_like(d)
    elif kernel == "metric":
        # Точка, совпавшая по времени с запросом, получила бы бесконечный вес.
        eps = 1e-9 * max(1.0, float(np.max(d)))
        w = np.maximum(d, eps) ** -power
    elif kernel == "exponential":
        # Сдвиг на минимальное расстояние: без него при малом bandwidth все веса уходят в ноль.
        w = np.exp(-(d**2 - np.min(d) ** 2) / (2.0 * bandwidth**2))
    else:
        raise ValueError(f"kernel должен быть одним из {KERNELS}, получено {kernel!r}")
    return w / w.sum()


def _line(t, y, w, t_query):
    """Взвешенная прямая и её прогноз в `t_query`.

    Возвращает (взвешенное среднее t, значение в нём, наклон, inflation). Прогноз линеен по y:
    ŷ = Σ c_i·y_i, поэтому при независимом шуме σ ошибка прогноза новой точки имеет
    стандартное отклонение σ·inflation, inflation = √(1 + Σ c_i²). По двум точкам прогноз
    втрое шумнее самой точки, по пяти — заметно тише; без этой поправки только что
    перестроенная модель принимала бы собственный шум за новую разладку.
    """
    w = w / w.sum()
    t_mean = w @ t
    y_mean = w @ y
    dt = t - t_mean
    sxx = w @ dt**2
    # Одна точка или все в один момент — наклон не определён, считаем уровень постоянным.
    span = np.max(t) - np.min(t)
    if sxx <= 1e-12 * max(span, 1.0) ** 2:
        slope = 0.0
        c = w
    else:
        slope = (w @ (dt * (y - y_mean))) / sxx
        c = w * (1.0 + (t_query - t_mean) * dt / sxx)
    return t_mean, y_mean, slope, float(np.sqrt(1.0 + c @ c))


@dataclass
class _Segment:
    """Окно точек текущего режима."""

    window: int
    t: deque = field(init=False)
    y: deque = field(init=False)
    sw: deque = field(init=False)

    def __post_init__(self):
        self.t = deque(maxlen=self.window)
        self.y = deque(maxlen=self.window)
        self.sw = deque(maxlen=self.window)

    def points(self):
        return list(zip(self.t, self.y, self.sw, strict=True))

    def append(self, t, y, sw):
        self.t.append(t)
        self.y.append(y)
        self.sw.append(sw)

    def copy(self):
        seg = _Segment(self.window)
        seg.t.extend(self.t)
        seg.y.extend(self.y)
        seg.sw.extend(self.sw)
        return seg

    def __len__(self):
        return len(self.t)


class WakefulWLS(RegressorMixin, BaseEstimator):
    """Взвешенная по времени линейная регрессия с защитой от засыпания.

    X — одна колонка времени (секунды, номера отсчётов и т. п.), y — наблюдаемая величина
    (например, потребление ОП, МБ). Точки обрабатываются по порядку времени, как поток;
    ``fit`` на всём ряде эквивалентен ``partial_fit`` по одной точке.

    Parameters
    ----------
    window : int, default=5
        Сколько последних точек текущего режима входит в регрессию. При ``adaptive_window=True`` —
        наибольшая допустимая длина окна (тогда её разумно взять побольше, например 30).
    kernel : {"metric", "exponential", "uniform"}, default="metric"
        Весовое ядро по расстоянию до момента прогноза.
    power : float, default=1.0
        Степень n метрического ядра.
    bandwidth : float, default=0.5
        Ширина h экспоненциального ядра (в единицах времени).
    guard : bool, default=True
        Включить защиту от засыпания. При ``False`` — базовая взвешенная регрессия по окну.
    threshold : float, default=4.0
        Порог разладки: ошибка прогноза новой точки в стандартных отклонениях ошибки прогноза.
        Шум оценивается по медиане модулей последних остатков; после разладки, пока у нового
        режима мало остатков, используется оценка прежнего режима.
    min_scale : float or "auto", default="auto"
        Нижняя граница оценки шума в единицах y. На идеально ровной полке оценка шума нулевая,
        и без границы разладкой считалось бы любое отклонение. ``"auto"`` — 1 % от среднего
        модуля y в окне. Для ОП разумно задать явно, например 5–10 МБ.
    scale_window : int, default=30
        Сколько последних остатков используется для оценки шума (медиана модулей).
    confirm_up : int, default=0
        Сколько точек должно подтвердить рост, прежде чем модель перестроится. 0 — сразу:
        недодать память опаснее, чем выдать лишнюю. Если следующая точка вернулась к старому
        режиму, перестройка откатывается.
    confirm_down : int, default=1
        То же для спада. До подтверждения модель держит прежний уровень, чтобы кратковременный
        провал не приводил к отъёму памяти.
    adaptive_window : bool, default=False
        Выбирать длину окна на каждом прогнозе правилом ICI среди 3, 5, 8, 12, … до ``window``
        точек текущего режима: берётся самое длинное окно, доверительный интервал прогноза
        которого (±``ici_gamma`` его стандартных отклонений) пересекается с интервалами всех
        более коротких окон.
    ici_gamma : float, default=1.5
        Ширина доверительных интервалов ICI. Меньше — окно короче и чувствительнее, больше —
        длиннее и тише.
    cusum_h : float or None, default=None
        Порог накопительного детектора CUSUM в стандартных отклонениях ошибки прогноза.
        ``None`` — детектор выключен. При срабатывании окно строится заново с точки, с которой
        начала копиться сумма.
    cusum_k : float, default=0.5
        Допуск CUSUM: ошибки меньше ``cusum_k`` стандартных отклонений не накапливаются.
        Обычно половина наименьшего сдвига, который нужно ловить.
    coverage : float, default=0.95
        Целевая доля точек внутри интервала ``predict_interval``.
    aci_rate : float, default=0.01
        Шаг онлайн-подстройки уровня интервала: после каждого промаха интервал расширяется, после
        попадания — понемногу сужается, так что фактическое покрытие держится около ``coverage``
        и после смены режима. 0 — без подстройки (обычный сплит-конформный интервал).
    interval_window : int, default=200
        Сколько последних нормированных ошибок прогноза используется для квантиля интервала.

    Attributes
    ----------
    coef_ : float
        Наклон прямой для прогноза на следующий шаг.
    intercept_ : float
        Свободный член той же прямой: прогноз = intercept_ + coef_ · t.
    change_points_ : list of float
        Моменты обнаруженных разладок (начала новых режимов).
    spikes_ : list of float
        Моменты точек, признанных кратковременными всплесками и отброшенных.
    interval_alpha_ : float
        Текущий подстроенный уровень промахов интервала (в начале — ``1 − coverage``).
    n_features_in_ : int
        Всегда 1.
    """

    def __init__(
        self,
        window=5,
        kernel="metric",
        power=1.0,
        bandwidth=0.5,
        guard=True,
        threshold=4.0,
        min_scale="auto",
        scale_window=30,
        confirm_up=0,
        confirm_down=1,
        adaptive_window=False,
        ici_gamma=1.5,
        cusum_h=None,
        cusum_k=0.5,
        coverage=0.95,
        aci_rate=0.01,
        interval_window=200,
    ):
        self.window = window
        self.kernel = kernel
        self.power = power
        self.bandwidth = bandwidth
        self.guard = guard
        self.threshold = threshold
        self.min_scale = min_scale
        self.scale_window = scale_window
        self.confirm_up = confirm_up
        self.confirm_down = confirm_down
        self.adaptive_window = adaptive_window
        self.ici_gamma = ici_gamma
        self.cusum_h = cusum_h
        self.cusum_k = cusum_k
        self.coverage = coverage
        self.aci_rate = aci_rate
        self.interval_window = interval_window

    # --- sklearn API -------------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Обучить модель на ряде заново. Точки сортируются по времени."""
        t, y, sw = self._validate(X, y, sample_weight, reset=True)
        self._check_params()
        order = np.argsort(t, kind="stable")
        self._init_state()
        self._consume(t[order], y[order], sw[order])
        return self

    def partial_fit(self, X, y, sample_weight=None):
        """Дообучить модель на новых точках. Время не должно идти назад."""
        first = not hasattr(self, "_segment")
        t, y, sw = self._validate(X, y, sample_weight, reset=first)
        if first:
            self._check_params()
            self._init_state()
        order = np.argsort(t, kind="stable")
        t, y, sw = t[order], y[order], sw[order]
        if self._last_t is not None and len(t) and t[0] < self._last_t:
            raise ValueError(
                f"partial_fit: время {t[0]} раньше последней точки {self._last_t}; используйте fit"
            )
        self._consume(t, y, sw)
        return self

    def predict(self, X):
        """Прогноз в моменты времени X по текущему окну."""
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.array([self._predict_one(self._segment, q) for q in X[:, 0]])

    def predict_interval(self, X):
        """Интервал прогноза (lower, upper) в моменты X с целевым покрытием ``coverage``.

        Полуширина — квантиль уровня ``1 − interval_alpha_`` нормированных модулей последних
        ошибок прогноза на шаг вперёд, умноженный на шум прогноза в данной точке. Пока ошибок
        меньше пяти, интервал бесконечен.
        """
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        prediction, half = np.array([self._interval_one(q) for q in X[:, 0]]).T
        return prediction - half, prediction + half

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags

    # --- внутреннее --------------------------------------------------------------------------

    def _validate(self, X, y, sample_weight, reset):
        X, y = validate_data(self, X, y, reset=reset, dtype=np.float64, y_numeric=True)
        if X.shape[1] != 1:
            raise ValueError(f"X должен содержать одну колонку времени, получено {X.shape[1]}")
        if sample_weight is None:
            sw = np.ones(len(y))
        else:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != y.shape:
                raise ValueError(f"sample_weight: форма {sw.shape}, ожидалась {y.shape}")
            if np.any(sw < 0) or not np.all(np.isfinite(sw)):
                raise ValueError("sample_weight должны быть конечными и неотрицательными")
        return X[:, 0], np.asarray(y, dtype=float), sw

    def _check_params(self):
        if not (isinstance(self.window, int | np.integer) and self.window >= 2):
            raise ValueError(f"window должен быть целым ≥ 2, получено {self.window!r}")
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel должен быть одним из {KERNELS}, получено {self.kernel!r}")
        for name, ok, rule in (
            ("power", self.power >= 0, "≥ 0"),
            ("bandwidth", self.bandwidth > 0, "> 0"),
            ("threshold", self.threshold > 0, "> 0"),
            ("ici_gamma", self.ici_gamma > 0, "> 0"),
            ("cusum_h", self.cusum_h is None or self.cusum_h > 0, "None или > 0"),
            ("cusum_k", self.cusum_k >= 0, "≥ 0"),
            ("coverage", 0 < self.coverage < 1, "в (0, 1)"),
            ("aci_rate", self.aci_rate >= 0, "≥ 0"),
        ):
            if not ok:
                raise ValueError(f"{name} должен быть {rule}, получено {getattr(self, name)!r}")
        if not (self.min_scale == "auto" or (np.isscalar(self.min_scale) and self.min_scale >= 0)):
            raise ValueError(f"min_scale должен быть 'auto' или числом ≥ 0, получено {self.min_scale!r}")
        if not (isinstance(self.scale_window, int | np.integer) and self.scale_window >= 1):
            raise ValueError(f"scale_window должен быть целым ≥ 1, получено {self.scale_window!r}")
        for name, low in (("confirm_up", 0), ("confirm_down", 0), ("interval_window", 1)):
            value = getattr(self, name)
            if not (isinstance(value, int | np.integer) and value >= low):
                raise ValueError(f"{name} должен быть целым ≥ {low}, получено {value!r}")

    def _init_state(self):
        self._segment = _Segment(self.window)
        # Нормированные модули остатков текущего режима и σ шума прошлого режима.
        self._residuals = deque(maxlen=self.scale_window)
        self._prior_sigma = None
        self._steps = deque(maxlen=self.window)
        self._last_t = None
        # Окно и оценка шума до немедленной перестройки — для отката, если это был всплеск.
        self._backup = None
        # Точки, ждущие подтверждения разладки, и её направление.
        self._pending = []
        self._pending_dir = 0
        # Последняя точка прежнего режима — опора для проверки излома тренда.
        self._anchor = None
        # Суммы CUSUM вверх и вниз и моменты, с которых они копятся.
        self._cusum = (0.0, 0.0, None, None)
        # Нормированные модули ошибок прогноза на шаг вперёд — для интервала.
        self._scores = deque(maxlen=self.interval_window)
        self.interval_alpha_ = 1.0 - self.coverage
        self.change_points_ = []
        self.spikes_ = []

    def _consume(self, t, y, sw):
        for ti, yi, swi in zip(t, y, sw, strict=True):
            if self._last_t is not None and ti > self._last_t:
                self._steps.append(ti - self._last_t)
            self._last_t = ti
            if swi > 0:
                self._update(float(ti), float(yi), float(swi))
        if len(self._segment) == 0:
            raise ValueError("sample_weight: all weights are zero — нет ни одной точки с положительным весом")
        step = float(np.median(self._steps)) if self._steps else 1.0
        t_next = self._last_t + step
        t_mean, y_mean, slope, _ = self._fit_line(self._segment, t_next)
        self.coef_ = float(slope)
        self.intercept_ = float(y_mean - slope * t_mean)

    def _fit_line(self, seg, t_query, scale=None):
        t = np.fromiter(seg.t, float, len(seg))
        y = np.fromiter(seg.y, float, len(seg))
        sw = np.fromiter(seg.sw, float, len(seg))
        if self.adaptive_window and len(t) > _ICI_MIN_WINDOW:
            if scale is None:
                scale = self._scale(self._residuals, self._prior_sigma)
            if scale is not None:
                return self._ici_line(t, y, sw, t_query, scale)
        return self._line_on(t, y, sw, t_query)

    def _line_on(self, t, y, sw, t_query):
        w = kernel_weights(t, t_query, self.kernel, self.power, self.bandwidth) * sw
        return _line(t, y, w, t_query)

    def _ici_line(self, t, y, sw, t_query, scale):
        """Прямая по самому длинному окну, чей прогноз согласуется со всеми более короткими.

        Короткое окно почти не смещено, но шумно; длинное — тихое, но после излома смещено.
        Пока смещение мало, доверительные интервалы прогнозов (±γ·σ·√Σc²) пересекаются; первое
        окно, интервал которого вышел из общего пересечения, уже захватило прежний режим.
        """
        n = len(t)
        lower, upper = -np.inf, np.inf
        best = None
        k = _ICI_MIN_WINDOW
        while True:
            line = self._line_on(t[-k:], y[-k:], sw[-k:], t_query)
            t_mean, y_mean, slope, inflation = line
            prediction = y_mean + slope * (t_query - t_mean)
            half = self.ici_gamma * scale * np.sqrt(max(inflation**2 - 1.0, 0.0))
            lower, upper = max(lower, prediction - half), min(upper, prediction + half)
            if lower > upper:
                break
            best = line
            if k == n:
                break
            k = min(n, max(k + 1, round(k * _ICI_RATIO)))
        return best

    def _predict_one(self, seg, t_query):
        return self._forecast(seg, t_query)[0]

    def _forecast(self, seg, t_query, scale=None):
        """Прогноз и коэффициент, во сколько раз его ошибка шумнее одной точки."""
        t_mean, y_mean, slope, inflation = self._fit_line(seg, t_query, scale)
        return y_mean + slope * (t_query - t_mean), inflation

    def _interval_one(self, t_query):
        prediction, inflation = self._forecast(self._segment, t_query)
        return prediction, self._half_width(inflation)

    def _half_width(self, inflation):
        """Конформная полуширина интервала: квантиль нормированных ошибок × шум прогноза."""
        n = len(self._scores)
        if n < _MIN_RESIDUALS:
            return np.inf
        # Поправка (n + 1)/n — стандартная для сплит-конформного квантиля по n ошибкам.
        level = (1.0 - self.interval_alpha_) * (n + 1) / n
        if level >= 1.0:
            q = max(self._scores)
        else:
            q = float(np.quantile(np.fromiter(self._scores, float, n), level, method="higher"))
        return max(q, self._floor()) * inflation

    def _record_score(self, residual, inflation):
        """Учесть ошибку прогноза на шаг вперёд и подстроить уровень интервала (ACI)."""
        if self.aci_rate > 0 and len(self._scores) >= _MIN_RESIDUALS:
            miss = abs(residual) > self._half_width(inflation)
            alpha = self.interval_alpha_ + self.aci_rate * ((1.0 - self.coverage) - miss)
            self.interval_alpha_ = min(max(alpha, 0.0), 1.0)
        self._scores.append(abs(residual) / inflation)

    def _scale(self, residuals, prior_sigma):
        """σ шума с нижней границей; None — оценить пока не по чему (начало ряда)."""
        if len(residuals) >= _MIN_RESIDUALS:
            sigma = _MAD_TO_SIGMA * float(np.median(residuals))
        elif prior_sigma is not None:
            sigma = prior_sigma
        else:
            return None
        return max(sigma, self._floor())

    def _floor(self):
        if self.min_scale == "auto":
            floor = 0.01 * float(np.mean(np.abs(self._segment.y)))
        else:
            floor = float(self.min_scale)
        return max(floor, 1e-12)

    def _reset(self, points, trigger_residuals, anchor=None):
        self.change_points_.append(points[0][0])
        self._anchor = anchor if anchor is not None else (self._segment.t[-1], self._segment.y[-1])
        self._cusum = (0.0, 0.0, None, None)
        self._segment = _Segment(self.window)
        for p in points:
            self._segment.append(*p)
        if len(self._residuals) >= _MIN_RESIDUALS:
            # Новый режим — новый шум. Старая оценка остаётся опорой, пока не накопится новая:
            # иначе после каждого сброса детектор слеп на несколько точек и проспал бы второй скачок.
            self._prior_sigma = _MAD_TO_SIGMA * float(np.median(self._residuals))
            self._residuals = deque(maxlen=self.scale_window)
        else:
            # Повторная разладка, пока новая оценка шума не набралась, — вероятно, новый режим
            # просто шумнее прежнего. Её остатки учитываются, чтобы каскад сбросов затух.
            self._residuals.extend(trigger_residuals)

    def _update(self, t, y, sw):
        point = (t, y, sw)
        if len(self._segment) == 0:
            self._segment.append(*point)
            return

        prediction, inflation = self._forecast(self._segment, t)
        residual = y - prediction
        self._record_score(residual, inflation)

        if self._backup is not None and self._rolled_back(point, residual):
            return

        normalized = abs(residual) / inflation
        scale = self._scale(self._residuals, self._prior_sigma) if self.guard else None

        # Без защиты или в начале ряда, пока шум не оценить, разладка не проверяется. Остатки
        # копятся и без защиты: по ним адаптивное окно оценивает шум.
        if scale is None:
            if len(self._segment) >= 2:
                self._residuals.append(normalized)
            self._segment.append(*point)
            return

        # По одной точке наклон не оценить, прогноз — её уровень. Точка, легшая на излом, —
        # продолжение разворота тренда, а не новый скачок: иначе модель перестраивалась бы на
        # каждой точке крутого тренда и вечно отставала на шаг. Точка, не легшая ни на уровень,
        # ни на излом, проверяется дальше как обычно и может оказаться новой разладкой.
        if len(self._segment) == 1 and self._on_kink(t, y, scale):
            self._segment.append(*point)
            return

        z = residual / (scale * inflation)
        direction = 1 if z > self.threshold else -1 if z < -self.threshold else 0

        if self._pending:
            if direction == self._pending_dir:
                self._pending.append((point, normalized))
                if len(self._pending) > self._confirm(direction):
                    points, normalized_residuals = zip(*self._pending, strict=True)
                    self._reset(points, normalized_residuals)
                    self._pending = []
                return
            # Отклонение не подтвердилось — это был кратковременный всплеск или провал.
            self.spikes_.extend(p[0][0] for p in self._pending)
            self._pending = []

        if direction == 0:
            self._segment.append(*point)
            self._residuals.append(normalized)
            if self.cusum_h is not None and len(self._segment) > 2:
                self._cusum_step(t, z)
        elif self._confirm(direction) == 0:
            residuals = deque(self._residuals, self.scale_window)
            self._backup = (self._segment.copy(), residuals, self._prior_sigma, self._cusum)
            self._reset([point], [normalized])
        else:
            self._pending = [(point, normalized)]
            self._pending_dir = direction

    def _rolled_back(self, point, residual):
        """Откат немедленной перестройки, если новая точка вернулась к прежнему режиму."""
        t, y, _ = point
        old, old_residuals, old_prior, old_cusum = self._backup
        self._backup = None
        old_scale = self._scale(old_residuals, old_prior)
        old_prediction, old_inflation = self._forecast(old, t, old_scale)
        old_residual = y - old_prediction
        if (
            old_scale is None
            or abs(old_residual) > self.threshold * old_scale * old_inflation
            or abs(old_residual) >= abs(residual)
        ):
            return False
        self.spikes_.extend(self._segment.t)
        self.change_points_.pop()
        self._segment = old
        self._segment.append(*point)
        self._residuals, self._prior_sigma, self._cusum = old_residuals, old_prior, old_cusum
        self._residuals.append(abs(old_residual) / old_inflation)
        return True

    def _cusum_step(self, t, z):
        """Накопить ошибку прогноза; при превышении порога — перестроить окно с начала сдвига."""
        up, down, up_start, down_start = self._cusum
        new_up = max(0.0, up + z - self.cusum_k)
        new_down = max(0.0, down - z - self.cusum_k)
        if up == 0.0 and new_up > 0.0:
            up_start = t
        if down == 0.0 and new_down > 0.0:
            down_start = t
        self._cusum = (new_up, new_down, up_start, down_start)
        if new_up > self.cusum_h:
            start = up_start
        elif new_down > self.cusum_h:
            start = down_start
        else:
            return
        points = self._segment.points()
        before = [p for p in points if p[0] < start]
        after = [p for p in points if p[0] >= start]
        if not before:
            # Сдвиг начался раньше всего окна — выбрасывать нечего, начать накопление заново.
            self._cusum = (0.0, 0.0, None, None)
            return
        self._reset(after, [], anchor=before[-1][:2])

    def _on_kink(self, t, y, scale):
        """Лежит ли точка на прямой от последней точки прежнего режима через точку перестройки."""
        if self._anchor is None:
            return False
        t_a, y_a = self._anchor
        t_1, y_1 = self._segment.t[0], self._segment.y[0]
        if t_1 <= t_a:
            return False
        r = (t - t_1) / (t_1 - t_a)
        predicted = y_1 + (y_1 - y_a) * r
        # Прогноз по двум зашумлённым точкам: дисперсия ошибки (1 + (1 + r)² + r²)·σ².
        inflation = np.sqrt(1.0 + (1.0 + r) ** 2 + r**2)
        return abs(y - predicted) <= self.threshold * scale * inflation

    def _confirm(self, direction):
        return self.confirm_up if direction > 0 else self.confirm_down
