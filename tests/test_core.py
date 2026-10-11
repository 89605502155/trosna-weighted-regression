import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.estimator_checks import parametrize_with_checks

from trosna_weighted_regression import WakefulWLS, kernel_weights
from trosna_weighted_regression.core import _line


def one_step_forecasts(model, y, start):
    """Прогноз на шаг вперёд по каждой точке, начиная со `start`, как в адаптивной системе."""
    t = np.arange(len(y), dtype=float)
    model = clone(model).fit(t[:start, None], y[:start])
    out = []
    for k in range(start, len(y)):
        out.append(model.predict([[t[k]]])[0])
        model.partial_fit([[t[k]]], [y[k]])
    return np.array(out)


FLAT = np.full(20, 1000.0)


# --- ядра и базовая модель из статьи -------------------------------------------------------------


def test_metric_kernel_matches_article():
    t = np.arange(5.0)
    w = kernel_weights(t, 5.0, "metric", power=1)
    expected = 1 / (5 - t)
    np.testing.assert_allclose(w, expected / expected.sum())


def test_exponential_kernel_matches_article_and_survives_small_bandwidth():
    t = np.arange(5.0)
    h = 2.0
    expected = np.exp(-((5 - t) ** 2) / (2 * h**2))
    np.testing.assert_allclose(kernel_weights(t, 5.0, "exponential", bandwidth=h), expected / expected.sum())
    w = kernel_weights(t, 5.0, "exponential", bandwidth=0.01)
    assert np.isfinite(w).all()
    assert w[-1] == pytest.approx(1.0)


def test_without_guard_is_plain_weighted_regression_on_last_window():
    rng = np.random.default_rng(0)
    t = np.arange(12.0)
    y = rng.normal(size=12)
    model = WakefulWLS(guard=False).fit(t[:, None], y)
    w = kernel_weights(t[-5:], 13.0)
    slope, intercept = np.polyfit(t[-5:], y[-5:], 1, w=np.sqrt(w))
    assert model.predict([[13.0]])[0] == pytest.approx(intercept + slope * 13.0)


def test_uniform_kernel_is_ols():
    t = np.arange(5.0)
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    model = WakefulWLS(kernel="uniform", guard=False).fit(t[:, None], y)
    slope, intercept = np.polyfit(t, y, 1)
    assert model.coef_ == pytest.approx(slope)
    assert model.intercept_ == pytest.approx(intercept)


# --- засыпание и защита --------------------------------------------------------------------------


def test_step_up_baseline_overshoots_guard_follows():
    y = np.r_[FLAT, np.full(10, 1500.0)]
    base = one_step_forecasts(WakefulWLS(guard=False), y, 10)[10:]
    guard = one_step_forecasts(WakefulWLS(min_scale=10), y, 10)[10:]
    assert base.max() > 1600  # «пляшет» выше новой полки
    np.testing.assert_allclose(guard[1:], 1500.0)
    # Первую точку ступеньки не предсказать никому; дальше базовая модель ошибается, защищённая — нет.
    assert np.abs(base[1:] - y[21:]).sum() > 400


def test_trend_reversal_is_caught_after_one_point():
    y = np.r_[1000 - 10 * np.arange(20.0), 810 + 100 * np.arange(1, 11.0)]
    base = one_step_forecasts(WakefulWLS(guard=False), y, 10)[10:]
    guard = one_step_forecasts(WakefulWLS(min_scale=10), y, 10)[10:]
    np.testing.assert_allclose(guard[2:], y[22:])
    assert np.abs(base[2:] - y[22:]).max() > 50
    model = WakefulWLS(min_scale=10).fit(np.arange(30.0)[:, None], y)
    assert model.change_points_ == [20.0]


def test_short_spike_is_rolled_back():
    y = np.r_[FLAT, [1500.0], np.full(9, 1000.0)]
    guard = one_step_forecasts(WakefulWLS(min_scale=10), y, 10)[10:]
    assert guard[1] == pytest.approx(1500.0)  # рост принимается сразу — память выдаётся
    np.testing.assert_allclose(guard[2:], 1000.0)  # после возврата всплеск забыт
    model = WakefulWLS(min_scale=10).fit(np.arange(30.0)[:, None], y)
    assert model.spikes_ == [20.0]
    assert model.change_points_ == []


def test_short_dip_is_ignored_and_step_down_needs_confirmation():
    dip = np.r_[FLAT, [500.0], np.full(9, 1000.0)]
    np.testing.assert_allclose(one_step_forecasts(WakefulWLS(min_scale=10), dip, 10), 1000.0)

    down = np.r_[FLAT, np.full(10, 500.0)]
    guard = one_step_forecasts(WakefulWLS(min_scale=10), down, 10)[10:]
    assert guard[1] == pytest.approx(1000.0)  # одна точка спада ещё не повод отнимать память
    np.testing.assert_allclose(guard[2:], 500.0)


def test_confirm_down_zero_reacts_immediately():
    down = np.r_[FLAT, np.full(10, 500.0)]
    guard = one_step_forecasts(WakefulWLS(min_scale=10, confirm_down=0), down, 10)[10:]
    np.testing.assert_allclose(guard[1:], 500.0)


def test_noise_does_not_trigger_resets():
    rng = np.random.default_rng(1)
    y = 2000 + rng.normal(scale=5.0, size=2000)
    model = WakefulWLS(min_scale=1.0).fit(np.arange(2000.0)[:, None], y)
    assert len(model.change_points_) + len(model.spikes_) <= 10


def test_step_on_noise_is_detected():
    rng = np.random.default_rng(2)
    y = np.r_[np.full(200, 2000.0), np.full(50, 2400.0)] + rng.normal(scale=5.0, size=250)
    model = WakefulWLS(min_scale=1.0).fit(np.arange(250.0)[:, None], y)
    assert 200.0 in model.change_points_
    assert model.predict([[250.0]])[0] == pytest.approx(2400, abs=30)


def test_forecast_noise_inflation_matches_ols_formula():
    # Для МНК по точкам 0..4 дисперсия ошибки прогноза в 5: σ²(1 + 1/n + (5 − 2)² / Σ(t − 2)²) = 2,1σ².
    t = np.arange(5.0)
    *_, inflation = _line(t, np.zeros(5), np.ones(5), 5.0)
    assert inflation == pytest.approx(np.sqrt(2.1))


@pytest.mark.parametrize("kernel", ["metric", "exponential"])
@pytest.mark.parametrize("pre_slope", [0.0, -3.0, 3.0])
@pytest.mark.parametrize("flat_len", [10, 300])
def test_large_jump_on_noise_is_tracked_after_one_point(kernel, pre_slope, flat_len):
    """Главное свойство: точка, сильно выбившаяся из тренда, сразу меняет прогноз."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        t = np.arange(flat_len + 20.0)
        true = 2000 + pre_slope * t + np.where(t >= flat_len, 600.0, 0.0)
        y = true + rng.normal(scale=5.0, size=len(t))
        kw = {"kernel": kernel, "min_scale": 1.0} | ({"bandwidth": 3.0} if kernel == "exponential" else {})
        guard = one_step_forecasts(WakefulWLS(**kw), y, 5)[flat_len - 5 :]
        base = one_step_forecasts(WakefulWLS(guard=False, **kw), y, 5)[flat_len - 5 :]
        err = np.abs(guard[1:] - true[flat_len + 1 :])
        assert err.max() < 30, seed  # шум 5 МБ, прогноз по 2–3 свежим точкам — до ~6σ
        assert np.abs(base[1:5] - true[flat_len + 1 : flat_len + 5]).max() > 100


def test_trend_reversal_on_noise_follows_new_slope_quickly():
    """Долгий спад, затем рост: первая же растущая точка разворачивает прогноз."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = np.arange(240.0)
        true = np.where(t < 200, 2000 - 3 * t, 1400 + 100 * (t - 200))
        y = true + rng.normal(scale=5.0, size=len(t))
        guard = one_step_forecasts(WakefulWLS(min_scale=1.0), y, 5)[195:]  # с t = 200
        base = one_step_forecasts(WakefulWLS(guard=False), y, 5)[195:]
        # t = 201 — первая растущая точка, её не предсказать; на t = 202 прогноз по одной свежей
        # точке — её уровень; с t = 203 защищённая модель идёт по новому наклону.
        assert np.abs(guard[3:] - true[203:]).max() < 40, seed
        assert np.abs(base[3:5] - true[203:205]).max() > 40, seed


def test_false_alarm_followed_by_real_jump_does_not_double_the_jump():
    # Шумовая точка чуть выше порога, следом настоящий скачок: прямая через эти две точки
    # удвоила бы скачок в прогнозе, вместо этого модель перестраивается со скачка.
    y = np.r_[np.full(20, 2000.0), [2030.0], np.full(10, 2600.0)]
    guard = one_step_forecasts(WakefulWLS(min_scale=5.0), y, 5)[15:]
    np.testing.assert_allclose(guard[7:], 2600.0)


def test_noisier_new_regime_does_not_cause_reset_cascade():
    rng = np.random.default_rng(6)
    y = np.r_[2000 + rng.normal(scale=1.0, size=300), 2600 + rng.normal(scale=40.0, size=200)]
    model = WakefulWLS(min_scale=1.0).fit(np.arange(500.0)[:, None], y)
    assert model.change_points_[0] == 300.0
    # Пока новая оценка шума не набралась, возможны несколько повторных перестроек, но не каскад.
    assert len(model.change_points_) <= 10
    assert len(model.change_points_) - (model.change_points_[-1] > 320) <= 10


def test_short_history_start_does_not_blow_up():
    rng = np.random.default_rng(7)
    true = np.r_[np.full(10, 2000.0), np.full(30, 2600.0)]
    y = true + rng.normal(scale=5.0, size=40)
    guard = one_step_forecasts(WakefulWLS(min_scale=1.0), y, 3)
    assert np.abs(guard[8:] - true[11:]).max() < 30


# --- адаптивное окно, CUSUM, интервал ------------------------------------------------------------

ADAPTIVE = {"window": 30, "adaptive_window": True, "min_scale": 1.0}


def test_adaptive_window_is_quieter_on_stationary_noise():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        y = 2000 + rng.normal(scale=5.0, size=400)
        adaptive = one_step_forecasts(WakefulWLS(**ADAPTIVE), y, 5)[45:]
        fixed = one_step_forecasts(WakefulWLS(min_scale=1.0), y, 5)[45:]
        assert np.abs(adaptive - 2000).mean() < 0.6 * np.abs(fixed - 2000).mean(), seed


def test_adaptive_window_shrinks_after_kink_without_guard():
    """Без детектора разладки длинное окно засыпает на изломе, ICI его укорачивает.

    Нагляднее всего на равных весах: метрическое ядро и само почти не смотрит на старые точки.
    """
    for seed in range(5):
        rng = np.random.default_rng(seed)
        t = np.arange(260.0)
        true = np.where(t < 200, 2000, 2000 + 10 * (t - 200))
        y = true + rng.normal(scale=5.0, size=len(t))
        adaptive = one_step_forecasts(WakefulWLS(kernel="uniform", guard=False, **ADAPTIVE), y, 5)[195:]
        fixed = one_step_forecasts(WakefulWLS(kernel="uniform", guard=False, window=30), y, 5)[195:]
        # С t = 206 по t = 229 фиксированное окно ещё тянет полку, адаптивное — уже на наклоне.
        assert np.abs(adaptive[6:30] - true[206:230]).max() < 30, seed
        assert np.abs(fixed[6:30] - true[206:230]).max() > 40, seed


def test_large_jump_is_tracked_after_one_point_with_adaptive_window():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        true = np.r_[np.full(300, 2000.0), np.full(20, 2600.0)]
        y = true + rng.normal(scale=5.0, size=len(true))
        guard = one_step_forecasts(WakefulWLS(**ADAPTIVE, cusum_h=5.0), y, 5)[295:]
        assert np.abs(guard[1:] - true[301:]).max() < 30, seed


def test_cusum_catches_slope_change_below_threshold():
    """Наклон 15 МБ/шаг после полки: разовая ошибка ниже порога, но копится."""
    with_cusum, without = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        t = np.arange(260.0)
        true = np.where(t < 200, 2000, 2000 + 15 * (t - 200))
        y = true + rng.normal(scale=5.0, size=len(t))
        model = WakefulWLS(**ADAPTIVE, cusum_h=5.0)
        a = np.abs(one_step_forecasts(model, y, 5)[195:] - true[200:]).mean()
        b = np.abs(one_step_forecasts(WakefulWLS(**ADAPTIVE), y, 5)[195:] - true[200:]).mean()
        assert a <= b + 1e-9, seed
        with_cusum.append(a)
        without.append(b)
    assert np.mean(with_cusum) < 0.85 * np.mean(without)


def test_cusum_restarts_window_from_start_of_drift():
    rng = np.random.default_rng(1)
    t = np.arange(260.0)
    y = np.where(t < 200, 2000, 2000 + 15 * (t - 200)) + rng.normal(scale=5.0, size=len(t))
    model = WakefulWLS(**ADAPTIVE, cusum_h=5.0).fit(t[:, None], y)
    assert [c for c in model.change_points_ if c > 100] == [200.0]


def test_cusum_does_not_fire_on_noise():
    rng = np.random.default_rng(1)
    y = 2000 + rng.normal(scale=5.0, size=2000)
    model = WakefulWLS(**ADAPTIVE, cusum_h=5.0).fit(np.arange(2000.0)[:, None], y)
    assert len(model.change_points_) <= 3


def interval_coverage(model, y, start):
    t = np.arange(len(y), dtype=float)
    model = clone(model).fit(t[:start, None], y[:start])
    inside = []
    for k in range(start, len(y)):
        lower, upper = model.predict_interval([[t[k]]])
        inside.append(lower[0] <= y[k] <= upper[0])
        model.partial_fit([[t[k]]], [y[k]])
    return np.array(inside)


def test_interval_has_target_coverage_on_heavy_tailed_noise():
    rng = np.random.default_rng(0)
    y = 2000 + 5.0 * rng.standard_t(3, size=1500)
    inside = interval_coverage(WakefulWLS(min_scale=1.0), y, 5)[100:]
    assert 0.93 <= inside.mean() <= 0.97


def test_interval_adapts_after_regime_changes():
    """Скачки и рост шума: онлайн-подстройка уровня возвращает покрытие к цели."""
    rng = np.random.default_rng(1)
    level = np.repeat([2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0], 150)
    noise = rng.normal(scale=5.0, size=900) * np.r_[np.ones(450), np.full(450, 3.0)]
    y = level + noise
    adaptive = interval_coverage(WakefulWLS(min_scale=1.0), y, 5)[100:].mean()
    static = interval_coverage(WakefulWLS(min_scale=1.0, aci_rate=0.0), y, 5)[100:].mean()
    assert abs(adaptive - 0.95) < 0.015
    assert abs(adaptive - 0.95) < abs(static - 0.95)


def test_interval_contains_prediction_and_is_infinite_on_short_history():
    rng = np.random.default_rng(2)
    t = np.arange(100.0)[:, None]
    y = 2000 + rng.normal(scale=5.0, size=100)
    model = WakefulWLS(min_scale=1.0).fit(t, y)
    lower, upper = model.predict_interval([[100.0], [101.0]])
    prediction = model.predict([[100.0], [101.0]])
    assert np.all(lower < prediction)
    assert np.all(prediction < upper)
    assert np.all(upper - lower < 40)
    short_lower, short_upper = WakefulWLS().fit(t[:3], y[:3]).predict_interval([[3.0]])
    assert np.isneginf(short_lower[0])
    assert np.isposinf(short_upper[0])


# --- поток и интерфейс ---------------------------------------------------------------------------


def test_partial_fit_matches_fit():
    rng = np.random.default_rng(3)
    y = np.r_[np.full(30, 100.0), np.full(30, 160.0)] + rng.normal(size=60)
    t = np.arange(60.0)[:, None]
    full = WakefulWLS(min_scale=0.5).fit(t, y)
    stream = WakefulWLS(min_scale=0.5)
    for k in range(0, 60, 7):
        stream.partial_fit(t[k : k + 7], y[k : k + 7])
    assert stream.change_points_ == full.change_points_
    assert stream.predict([[60.0]]) == pytest.approx(full.predict([[60.0]]))


def test_partial_fit_matches_fit_with_all_mechanisms():
    rng = np.random.default_rng(3)
    y = np.r_[np.full(80, 100.0), 100 + 2 * np.arange(80.0)] + rng.normal(size=160)
    t = np.arange(160.0)[:, None]
    params = {"window": 30, "adaptive_window": True, "cusum_h": 5.0, "min_scale": 0.5}
    full = WakefulWLS(**params).fit(t, y)
    stream = WakefulWLS(**params)
    for k in range(0, 160, 9):
        stream.partial_fit(t[k : k + 9], y[k : k + 9])
    assert stream.change_points_ == full.change_points_
    assert stream.interval_alpha_ == pytest.approx(full.interval_alpha_)
    assert stream.predict_interval([[160.0]]) == pytest.approx(full.predict_interval([[160.0]]))


def test_partial_fit_rejects_past_time():
    model = WakefulWLS().partial_fit([[5.0]], [1.0])
    with pytest.raises(ValueError, match="раньше"):
        model.partial_fit([[4.0]], [1.0])


def test_fit_sorts_by_time_and_zero_weight_ignores_point():
    t = np.arange(10.0)
    y = np.r_[np.full(9, 1.0), 50.0]
    perm = np.random.default_rng(4).permutation(10)
    a = WakefulWLS(guard=False).fit(t[perm, None], y[perm])
    b = WakefulWLS(guard=False).fit(t[:, None], y)
    assert a.predict([[10.0]]) == pytest.approx(b.predict([[10.0]]))
    c = WakefulWLS(guard=False).fit(t[:, None], y, sample_weight=np.r_[np.ones(9), 0.0])
    assert c.predict([[10.0]])[0] == pytest.approx(1.0)


def test_rejects_several_features_and_bad_params():
    with pytest.raises(ValueError, match="одну колонку"):
        WakefulWLS().fit(np.ones((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="window"):
        WakefulWLS(window=1).fit(np.arange(5.0)[:, None], np.ones(5))
    with pytest.raises(ValueError, match="kernel"):
        WakefulWLS(kernel="gauss").fit(np.arange(5.0)[:, None], np.ones(5))
    for bad in (
        {"ici_gamma": 0},
        {"cusum_h": -1.0},
        {"cusum_k": -0.1},
        {"coverage": 1.0},
        {"aci_rate": -0.01},
        {"interval_window": 0},
    ):
        with pytest.raises(ValueError, match=next(iter(bad))):
            WakefulWLS(**bad).fit(np.arange(5.0)[:, None], np.ones(5))


def test_works_in_grid_search_and_pipeline():
    rng = np.random.default_rng(5)
    t = np.arange(120.0)[:, None]
    y = np.sin(t[:, 0] / 10) * 100 + rng.normal(size=120)
    search = GridSearchCV(
        WakefulWLS(), {"power": [0.5, 1, 2], "window": [3, 5, 8]}, cv=TimeSeriesSplit(3)
    ).fit(t, y)
    assert set(search.best_params_) == {"power", "window"}
    pipe = make_pipeline(FunctionTransformer(lambda x: x - 60), WakefulWLS()).fit(t, y)
    assert pipe.predict(t[-1:]).shape == (1,)


# Эти проверки sklearn подают X из нескольких признаков, а модель принимает ровно одну
# колонку времени и падает на проверке формы. Все остальные стандартные проверки проходят.
MULTI_FEATURE_CHECKS = (
    "check_fit_score_takes_y",
    "check_estimators_overwrite_params",
    "check_dont_overwrite_parameters",
    "check_estimators_fit_returns_self",
    "check_readonly_memmap_input",
    "check_n_features_in_after_fitting",
    "check_positive_only_tag_during_fit",
    "check_estimators_dtypes",
    "check_sample_weights_not_an_array",
    "check_sample_weights_list",
    "check_all_zero_sample_weights_error",
    "check_sample_weights_shape",
    "check_sample_weights_not_overwritten",
    "check_sample_weight_equivalence_on_dense_data",
    "check_dtype_object",
    "check_pipeline_consistency",
    "check_estimators_nan_inf",
    "check_estimators_pickle",
    "check_f_contiguous_array_estimator",
    "check_regressors_train",
    "check_regressor_data_not_an_array",
    "check_estimators_partial_fit_n_features",
    "check_regressors_no_decision_function",
    "check_supervised_y_2d",
    "check_regressors_int",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_fit2d_1sample",
    "check_dict_unchanged",
    "check_fit_idempotent",
    "check_fit_check_is_fitted",
    "check_n_features_in",
    "check_fit2d_predict1d",
)


@parametrize_with_checks(
    [WakefulWLS(), WakefulWLS(window=30, adaptive_window=True, cusum_h=5.0)],
    expected_failed_checks=lambda est: dict.fromkeys(MULTI_FEATURE_CHECKS, "одна колонка времени"),
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
