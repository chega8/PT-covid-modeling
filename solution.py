#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для анализа динамики COVID-19 с использованием стохастической генеративной модели.

Этот модуль реализует стохастическую генеративную модель для анализа динамики COVID-19
с использованием библиотеки PyMC (версия > 3). Модель оценивает эффективное репродуктивное 
число R(t) и прогнозирует количество новых случаев заболевания.

Основные функции:
1. Загрузка и предобработка данных COVID-19 из OWID
2. Построение стохастической модели с использованием PyMC
3. Обучение модели на исторических данных
4. Прогнозирование динамики заболеваемости
5. Визуализация результатов

Страны для анализа: Россия, Италия, Германия, Франция
Период обучения: 01.01.2020 - 01.12.2020 (с исключением дней до достижения 100 случаев в день)
Период прогноза: 02.12.2020 - 14.12.2020
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Настройка окружения для PyTensor (необходимо для корректной работы с BLAS)
os.environ.setdefault('PYTENSOR_FLAGS', 'blas__ldflags=')

# Подавление предупреждений для чистого вывода
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist

import pymc as pm
import pytensor
import pytensor.tensor as pt

try:
    import arviz as az  # noqa
except Exception:
    az = None

# =============================================================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# =============================================================================

# Пути к данным
OWID_LOCAL_FILE = "owid-covid-data.csv"

# Временные периоды анализа
TRAIN_START = "2020-01-01"
TRAIN_END   = "2020-12-01"
FORECAST_START = "2020-12-02"
FORECAST_END   = "2020-12-14"

# Страны для анализа
COUNTRY_NAMES = {"Russia": "Russia", "Italy": "Italy", "Germany": "Germany", "France": "France"}

# Пороговые значения для модели
MIN_CASES_THRESHOLD = 100  # Минимальное количество случаев в день для начала анализа
SERIAL_MEAN = 4.0         # Среднее значение серийного интервала (дни)
SERIAL_SD   = 2.0         # Стандартное отклонение серийного интервала
MAX_LAG     = 15          # Максимальная задержка для учета инфекционности

# Параметры MCMC сэмплирования
DRAWS = 3000              # Количество сэмплов для генерации
TUNE  = 2000              # Количество сэмплов для настройки
CHAINS = 4                # Количество цепей MCMC
TARGET_ACCEPT = 0.97       # Целевая вероятность принятия для NUTS
RANDOM_SEED = 42          # Семя для воспроизводимости результатов


@dataclass
class ModelConfig:
    """
    Конфигурация параметров стохастической модели COVID-19.
    
    Этот класс содержит все параметры, необходимые для построения и обучения
    стохастической модели динамики COVID-19.
    
    Attributes:
        max_lag (int): Максимальная задержка для учета инфекционности (дни)
        serial_mean (float): Среднее значение серийного интервала (дни)
        serial_sd (float): Стандартное отклонение серийного интервала (дни)
        target_accept (float): Целевая вероятность принятия для NUTS сэмплера
        draws (int): Количество сэмплов для генерации
        tune (int): Количество сэмплов для настройки сэмплера
        chains (int): Количество цепей MCMC
        random_seed (int): Семя для воспроизводимости результатов
    """
    max_lag: int = MAX_LAG
    serial_mean: float = SERIAL_MEAN
    serial_sd: float = SERIAL_SD
    target_accept: float = TARGET_ACCEPT
    draws: int = DRAWS
    tune: int = TUNE
    chains: int = CHAINS
    random_seed: int = RANDOM_SEED


# =============================================================================
# УТИЛИТЫ ДЛЯ СТАТИСТИЧЕСКОГО АНАЛИЗА
# =============================================================================

def _hdi_1d(sample: np.ndarray, prob: float = 0.9) -> Tuple[float, float]:
    """
    Вычисляет Highest Density Interval (HDI) для одномерного распределения.
    
    HDI - это интервал, который содержит заданную долю вероятности распределения
    и имеет наименьшую ширину среди всех таких интервалов.
    
    Args:
        sample (np.ndarray): Массив сэмплов из распределения
        prob (float): Доля вероятности, которую должен содержать интервал (по умолчанию 0.9)
        
    Returns:
        Tuple[float, float]: Нижняя и верхняя границы HDI интервала
        
    Note:
        Если массив пустой, возвращает (nan, nan)
    """
    x = np.sort(np.asarray(sample).ravel())
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    
    # Вычисляем количество точек, которые должны попасть в интервал
    m = max(1, int(np.floor(prob * n)))
    
    # Находим интервал минимальной ширины, содержащий m точек
    widths = x[m:] - x[: n - m]
    j = int(np.argmin(widths))
    
    return float(x[j]), float(x[j + m])


def hdi(samples: np.ndarray, prob: float = 0.9, axis: int = 0) -> np.ndarray:
    """
    Вычисляет Highest Density Interval (HDI) для многомерного распределения.
    
    Эта функция применяет HDI к каждому измерению многомерного массива сэмплов,
    что полезно для анализа результатов MCMC сэмплирования.
    
    Args:
        samples (np.ndarray): Массив сэмплов из многомерного распределения
        prob (float): Доля вероятности для HDI интервала (по умолчанию 0.9)
        axis (int): Ось, по которой группируются сэмплы (по умолчанию 0)
        
    Returns:
        np.ndarray: Массив формы (..., 2), где последняя ось содержит
                   [нижняя_граница, верхняя_граница] для каждого измерения
                   
    Example:
        >>> samples = np.random.normal(0, 1, (1000, 10))  # 1000 сэмплов, 10 параметров
        >>> hdi_intervals = hdi(samples, prob=0.95)
        >>> print(hdi_intervals.shape)  # (10, 2)
    """
    samples = np.asarray(samples)
    samples = np.moveaxis(samples, axis, 0)
    out_low = []
    out_high = []
    rest_shape = samples.shape[1:]
    flat = samples.reshape(samples.shape[0], -1)
    
    # Применяем HDI к каждому измерению
    for k in range(flat.shape[1]):
        lo, hi = _hdi_1d(flat[:, k], prob)
        out_low.append(lo)
        out_high.append(hi)
    
    out_low = np.array(out_low).reshape(rest_shape)
    out_high = np.array(out_high).reshape(rest_shape)
    return np.stack([out_low, out_high], axis=-1)

# =============================================================================
# МОДУЛЬ ДЛЯ РАБОТЫ С ДАННЫМИ
# =============================================================================

def ensure_dir(path: str) -> None:
    """
    Создает директорию, если она не существует.
    
    Эта функция проверяет существование указанного пути и создает директорию
    со всеми необходимыми родительскими директориями, если она не существует.
    
    Args:
        path (str): Путь к директории, которую нужно создать
        
    Note:
        Использует os.makedirs с exist_ok=True для безопасного создания директорий
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def date_range_days(start: str, end: str) -> int:
    """
    Вычисляет количество дней между двумя датами (включительно).
    
    Args:
        start (str): Начальная дата в формате 'YYYY-MM-DD'
        end (str): Конечная дата в формате 'YYYY-MM-DD'
        
    Returns:
        int: Количество дней между датами (включительно)
        
    Example:
        >>> date_range_days("2020-01-01", "2020-01-03")
        3
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return (e - s).days + 1


def load_owid_data() -> pd.DataFrame:
    """
    Загружает данные COVID-19 из Our World in Data (OWID).
    
    Эта функция сначала пытается загрузить данные из локального файла.
    Если локальный файл не найден, она загружает данные с официального сайта OWID
    и сохраняет их локально для последующего использования.
    
    Returns:
        pd.DataFrame: DataFrame с данными COVID-19, содержащий колонки:
                     - date: дата (datetime)
                     - location: название страны/региона
                     - new_cases: количество новых случаев в день
                     - и другие метрики COVID-19
                     
    Note:
        Данные автоматически кэшируются в локальный файл для ускорения последующих загрузок
    """
    if os.path.exists(OWID_LOCAL_FILE):
        print(f"[data] Loading local '{OWID_LOCAL_FILE}'")
        return pd.read_csv(OWID_LOCAL_FILE, parse_dates=["date"])
    else:
        print(f"[data] Local '{OWID_LOCAL_FILE}' not found. Attempting to download...")
        df = pd.read_csv(OWID_URL, parse_dates=["date"])
        df.to_csv(OWID_LOCAL_FILE, index=False)
        print(f"[data] Downloaded and cached to '{OWID_LOCAL_FILE}'")
        return df

def prepare_country_series(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Подготавливает временной ряд данных COVID-19 для конкретной страны.
    
    Эта функция выполняет следующие операции:
    1. Фильтрует данные по указанной стране
    2. Сортирует данные по дате
    3. Обрезает данные по заданному временному окну
    4. Очищает данные (заполняет пропуски, обрезает отрицательные значения)
    5. Определяет начальную точку анализа (когда количество случаев превышает порог)
    6. Возвращает очищенный временной ряд
    
    Args:
        df (pd.DataFrame): Полный датасет COVID-19 из OWID
        country (str): Название страны для анализа
        
    Returns:
        pd.DataFrame: Очищенный временной ряд с колонками:
                     - date: дата (datetime)
                     - new_cases: количество новых случаев в день (float64)
                     
    Raises:
        ValueError: Если страна не достигла порогового значения случаев в период обучения
        
    Note:
        Начальная точка анализа определяется как первый день, когда количество
        новых случаев превысило MIN_CASES_THRESHOLD в период обучения
    """
    # Фильтруем данные по стране и оставляем только необходимые колонки
    sub = df.loc[df["location"] == country, ["date", "new_cases"]].copy()
    
    # Сортируем по дате и сбрасываем индексы
    sub = sub.sort_values("date").reset_index(drop=True)
    
    # Определяем общее временное окно (обучение + прогноз)
    overall_start = pd.to_datetime(TRAIN_START)
    overall_end = pd.to_datetime(FORECAST_END)
    
    # Обрезаем данные по временному окну
    sub = sub[(sub["date"] >= overall_start) & (sub["date"] <= overall_end)].copy()
    
    # Очищаем данные: заполняем пропуски нулями и обрезаем отрицательные значения
    sub["new_cases"] = sub["new_cases"].fillna(0.0).clip(lower=0.0)
    
    # Определяем период обучения для поиска порогового значения
    train_mask = (sub["date"] >= pd.to_datetime(TRAIN_START)) & (sub["date"] <= pd.to_datetime(TRAIN_END))
    train_sub = sub.loc[train_mask].copy()
    
    # Находим первый день, когда количество случаев превысило порог
    idx_threshold = train_sub.index[train_sub["new_cases"] >= MIN_CASES_THRESHOLD]
    if len(idx_threshold) == 0:
        raise ValueError(f"{country}: never reached {MIN_CASES_THRESHOLD} daily cases in training window.")
    
    start_idx = idx_threshold[0]
    start_date = sub.loc[start_idx, "date"]
    print(f"[{country}] Using training start at threshold-crossing date: {start_date.date()}")
    
    # Обрезаем данные, начиная с порогового дня
    sub = sub[sub["date"] >= start_date].reset_index(drop=True)
    return sub

# =============================================================================
# МОДУЛЬ ДЛЯ РАБОТЫ С МОДЕЛЯМИ
# =============================================================================

def discrete_gamma_pmf(max_lag: int, mean: float, sd: float) -> np.ndarray:
    """
    Вычисляет дискретную функцию массы вероятности (PMF) для гамма-распределения.
    
    Эта функция создает дискретизированную версию гамма-распределения для моделирования
    серийного интервала (времени между заражением и передачей инфекции).
    Серийный интервал критически важен для оценки репродуктивного числа R(t).
    
    Args:
        max_lag (int): Максимальная задержка в днях (длина массива весов)
        mean (float): Среднее значение серийного интервала (дни)
        sd (float): Стандартное отклонение серийного интервала (дни)
        
    Returns:
        np.ndarray: Массив весов формы (max_lag,), представляющий PMF
                   для задержек от 1 до max_lag дней
                   
    Note:
        Веса нормализованы так, что их сумма равна 1.
        Используется для вычисления инфекционности в каждый момент времени.
        
    Mathematical Details:
        Параметры гамма-распределения:
        - shape parameter k = (mean/sd)²
        - scale parameter θ = sd²/mean
        
        PMF вычисляется как разность CDF в точках s+0.5 и s-0.5
    """
    # Вычисляем параметры гамма-распределения
    k = (mean / sd) ** 2      # shape parameter
    theta = (sd ** 2) / mean  # scale parameter
    
    # Создаем объект гамма-распределения
    gamma_dist_obj = gamma_dist(a=k, scale=theta)
    cdf = gamma_dist_obj.cdf
    
    # Вычисляем дискретную PMF для задержек от 1 до max_lag дней
    w = np.array([
        max(cdf(s + 0.5) - cdf(s - 0.5), 0.0) 
        for s in range(1, max_lag + 1)
    ], dtype=float)
    
    # Нормализуем веса (сумма должна быть равна 1)
    w = w / max(w.sum(), 1e-12)
    
    return w

def _sample_with_fallback(config: ModelConfig):
    """
    Создает функцию сэмплирования с fallback механизмом.
    
    Эта функция возвращает функцию сэмплирования, которая сначала пытается использовать
    стандартный NUTS сэмплер PyMC, а в случае неудачи переключается на JAX/NumPyro NUTS.
    Это обеспечивает совместимость с различными версиями PyMC и доступными бэкендами.
    
    Args:
        config (ModelConfig): Конфигурация модели с параметрами сэмплирования
        
    Returns:
        callable: Функция сэмплирования, которая возвращает trace объект
        
    Note:
        Fallback на JAX/NumPyro требует установки дополнительных зависимостей:
        pip install jax jaxlib numpyro
    """
    def _s():
        try:
            # Попытка использовать стандартный NUTS сэмплер
            return pm.sample(
                draws=config.draws, 
                tune=config.tune, 
                chains=config.chains,
                random_seed=config.random_seed, 
                target_accept=config.target_accept,
                progressbar=True, 
                compute_convergence_checks=True
            )
        except Exception:
            # Fallback на JAX/NumPyro NUTS при неудаче стандартного сэмплера
            print("[warn] Standard NUTS failed; falling back to JAX/NumPyro NUTS (needs jax).")
            print(traceback.format_exc())
            return pm.sampling_jax.sample_numpyro_nuts(
                draws=config.draws, 
                tune=config.tune,
                chains=config.chains, 
                target_accept=config.target_accept,
                random_seed=config.random_seed, 
                chain_method="vectorized",
                postprocessing_backend="cpu"
            )
    return _s

def build_rt_model(y_obs: np.ndarray, forecast_horizon: int, config: ModelConfig) -> Tuple[pm.Model, Dict[str, int], np.ndarray]:
    """
    Строит стохастическую модель для оценки репродуктивного числа R(t).
    
    Эта функция создает байесовскую модель в PyMC для анализа динамики COVID-19.
    Модель включает:
    1. Стохастическое моделирование R(t) как случайного блуждания
    2. Учет инфекционности через серийный интервал
    3. Негативно-биномиальное распределение для количества случаев
    4. Возможность прогнозирования на заданный горизонт
    
    Mathematical Model:
        log(R(t)) = log(R(t-1)) + ε(t), где ε(t) ~ Normal(0, σ_rw)
        μ(t) = R(t) * Σ[w(s) * y(t-s)] для s от 1 до max_lag
        y(t) ~ NegativeBinomial(μ(t), α)
        
    Args:
        y_obs (np.ndarray): Наблюдаемые данные о количестве новых случаев (обучение)
        forecast_horizon (int): Горизонт прогноза в днях
        config (ModelConfig): Конфигурация модели
        
    Returns:
        Tuple[pm.Model, Dict[str, int], np.ndarray]: 
            - model: PyMC модель
            - lengths: Словарь с размерами временных окон
            - w: Веса серийного интервала
            
    Note:
        Модель использует логарифмическое представление R(t) для обеспечения положительности
        и стабильности численных вычислений.
    """
    # Определяем размеры временных окон
    T_train = len(y_obs)                    # Количество дней обучения
    H = int(forecast_horizon)               # Горизонт прогноза
    T_total = T_train + H                   # Общее количество дней
    
    # Вычисляем веса серийного интервала
    w = discrete_gamma_pmf(config.max_lag, config.serial_mean, config.serial_sd).astype(np.float64)
    
    # Вычисляем инфекционность для каждого дня обучения
    # Инфекционность = взвешенная сумма случаев за предыдущие дни
    infectiousness_train = np.zeros(T_train, dtype=np.float64)
    for t in range(T_train):
        s_max = min(config.max_lag, t)  # Максимальная доступная задержка
        if s_max >= 1:
            # Берем случаи за предыдущие дни
            prev = y_obs[t - np.arange(1, s_max + 1)]
            # Вычисляем взвешенную сумму
            infectiousness_train[t] = float((prev * w[:s_max]).sum())
    
    # Преобразуем в тензор PyTensor для использования в модели
    infectiousness_train_tt = pt.as_tensor_variable(infectiousness_train)
    
    # Строим байесовскую модель
    with pm.Model() as model:
        # Стандартное отклонение случайного блуждания для log(R(t))
        sigma_rw  = pm.HalfNormal("sigma_rw", sigma=0.07)

        # Начальное значение log(R(t))
        log_rt0 = pm.Normal("log_rt0", mu=np.log(1.1), sigma=0.3)

        # sigma_rw  = pm.HalfNormal("sigma_rw", sigma=0.07)   # train
        # sigma_fore = pm.HalfNormal("sigma_fore", sigma=0.02)  # forecast намного тише
        # nu_rw = pm.Exponential("nu_rw", 1/10)

        # # формируем вектор сигм по шагам:
        # sigmas = pt.concatenate([
        #     pt.repeat(sigma_rw, repeats=T_train - 1),
        #     pt.repeat(sigma_fore, repeats=H)                # последние H шагов — форкаст
        # ])

        # Шоки случайного блуждания
        eps = pm.Normal("eps", mu=0.0, sigma=sigma_rw, shape=T_total - 1)
        # eps = pm.StudentT("eps", nu=nu_rw, mu=0.0, sigma=sigma_rw, shape=T_total - 1)
        # eps = pm.StudentT("eps", nu=nu_rw, mu=0.0, sigma=sigmas, shape=T_total - 1)

        # Построение траектории log(R(t)) через кумулятивную сумму
        log_rt = pm.Deterministic("log_rt", 
                                 pt.concatenate([log_rt0.reshape((1,)), log_rt0 + pt.cumsum(eps)]))
        
        # Преобразование обратно в R(t)
        Rt = pm.Deterministic("Rt", pt.exp(log_rt))
        
        # Ожидаемое количество случаев в день обучения
        mu_train = Rt[:T_train] * infectiousness_train_tt
        
        # Параметр дисперсии для негативно-биномиального распределения
        # alpha = pm.HalfNormal("alpha", sigma=10.0)
        log_alpha = pm.Normal("log_alpha", mu=np.log(60.0), sigma=0.5)
        alpha = pm.Deterministic("alpha", pt.exp(log_alpha))
        # Наблюдаемые данные следуют негативно-биномиальному распределению
        _ = pm.NegativeBinomial("y_obs", 
                               mu=mu_train + 1e-6, 
                               alpha=alpha + 1e-6, 
                               observed=y_obs)
    
    # Возвращаем модель и вспомогательную информацию
    lengths = {"T_train": T_train, "T_total": T_total, "H": H}
    return model, lengths, w

# =============================================================================
# МОДУЛЬ ДЛЯ ПРОГНОЗИРОВАНИЯ И АНАЛИЗА
# =============================================================================

def fit_and_forecast_country(country: str, full_df: pd.DataFrame, outdir: str, config: ModelConfig, run_fit: bool = True) -> Dict[str, Any]:
    """
    Обучает модель и выполняет прогнозирование для конкретной страны.
    
    Эта функция является основным рабочим процессом для анализа COVID-19:
    1. Подготавливает данные для страны
    2. Строит стохастическую модель R(t)
    3. Обучает модель на исторических данных (если run_fit=True)
    4. Выполняет прогнозирование на заданный горизонт
    5. Вычисляет метрики качества прогноза
    6. Сохраняет результаты и создает визуализации
    
    Args:
        country (str): Название страны для анализа
        full_df (pd.DataFrame): Полный датасет COVID-19 из OWID
        outdir (str): Директория для сохранения результатов
        config (ModelConfig): Конфигурация модели
        run_fit (bool): Выполнять ли обучение модели (по умолчанию True)
        
    Returns:
        Dict[str, Any]: Словарь с результатами анализа, содержащий:
            - country: название страны
            - lengths: размеры временных окон
            - train_start/train_end: даты начала и конца обучения
            - forecast_start/forecast_end: даты прогноза
            - serial_pmf: веса серийного интервала
            - metrics: метрики качества (MAE, MAPE)
            - summary_files: пути к файлам с результатами
            - fig_files: пути к файлам с графиками
            - trace: объект trace из MCMC (если run_fit=True)
            
    Note:
        Если run_fit=False, модель строится, но не обучается. Это полезно для
        быстрой проверки корректности построения модели.
    """
    # Создаем выходную директорию
    ensure_dir(outdir)
    
    # Подготавливаем данные для страны
    country_df = prepare_country_series(full_df, country)
    
    # Определяем временные границы
    train_end_date = pd.to_datetime(TRAIN_END)
    forecast_start = pd.to_datetime(FORECAST_START)
    forecast_end = pd.to_datetime(FORECAST_END)
    
    # Разделяем данные на обучение и полное окно
    train_df = country_df[(country_df["date"] <= train_end_date)].copy().reset_index(drop=True)
    full_window_df = country_df.copy().reset_index(drop=True)
    
    # Извлекаем данные для обучения
    y_train = train_df["new_cases"].to_numpy().astype(np.float64)
    
    # Определяем горизонт прогноза
    H = date_range_days(FORECAST_START, FORECAST_END)
    
    # Строим модель
    model, lengths, w = build_rt_model(y_train, H, config)
    
    # Инициализируем структуру результатов
    results = {
        "country": country,
        "lengths": lengths,
        "train_start": train_df["date"].iloc[0] if not train_df.empty else None,
        "train_end": train_df["date"].iloc[-1] if not train_df.empty else None,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "serial_pmf": w,
        "metrics": {},
        "summary_files": {},
        "fig_files": {},
        "trace": None
    }
    
    # Пропускаем обучение, если не требуется
    if not run_fit:
        print(f"[{country}] Skipping sampling (run_fit=False). Model built successfully.")
        return results
    
    # Обучаем модель с помощью MCMC
    with model:
        trace = _sample_with_fallback(config)()
        # Генерируем дополнительные сэмплы для прогнозирования
        _ = pm.sample_posterior_predictive(trace, var_names=["Rt"], random_seed=config.random_seed)
    
    results["trace"] = trace
    
    # =============================================================================
    # БЛОК ПРОГНОЗИРОВАНИЯ
    # =============================================================================
    
    # Извлекаем сэмплы R(t) и параметра дисперсии из MCMC trace
    rt_arr = trace.posterior["Rt"].stack(sample=("chain", "draw")).values  # (time, samples)
    alpha_arr = trace.posterior["alpha"].stack(sample=("chain", "draw")).values  # (samples,)
    
    # Определяем размер окна для истории случаев
    S = config.max_lag
    
    # Подготавливаем историю случаев для прогнозирования
    if lengths["T_train"] >= S:
        # Если данных достаточно, берем последние S дней
        hist0 = train_df["new_cases"].to_numpy()[-S:].astype(float)
    else:
        # Если данных мало, дополняем нулями в начале
        hist0 = np.pad(train_df["new_cases"].to_numpy().astype(float), (S - lengths["T_train"], 0))
    
    # Инициализируем генератор случайных чисел для воспроизводимости
    rng = np.random.default_rng(config.random_seed)
    
    # Определяем параметры для прогнозирования
    H = lengths["H"]  # Горизонт прогноза
    W = w            # Веса серийного интервала
    
    def nb_draw(mu: float, alpha: float) -> int:
        """
        Генерирует случайное число из негативно-биномиального распределения.
        
        Args:
            mu (float): Среднее значение распределения
            alpha (float): Параметр дисперсии
            
        Returns:
            int: Случайное число случаев
        """
        n = np.clip(alpha, 1e-8, 1e8)
        p = n / (n + np.clip(mu, 1e-12, 1e12))
        return rng.negative_binomial(n=n, p=p)
    
    # Обрабатываем случай одномерного массива R(t)
    if rt_arr.ndim == 1:
        rt_arr = rt_arr.reshape(-1, 1)
    
    T_total, NS = rt_arr.shape  # NS - количество сэмплов
    
    # Генерируем прогнозы для каждого сэмпла
    y_future_samples = np.zeros((NS, H), dtype=float)
    for s_idx in range(NS):
        # Извлекаем траекторию R(t) для данного сэмпла
        rt_path = rt_arr[:, s_idx]
        rt_fore = rt_path[lengths["T_train"]: lengths["T_train"] + H]  # R(t) для прогноза
        
        # Извлекаем параметр дисперсии для данного сэмпла
        alpha_s = float(alpha_arr.flatten()[s_idx % alpha_arr.size])
        
        # Инициализируем историю случаев
        hist = hist0.copy()
        
        # Генерируем прогноз по дням
        for t in range(H):
            # Вычисляем инфекционность на основе истории
            infectious = float((hist * W).sum())
            
            # Вычисляем ожидаемое количество случаев
            mu_t = max(rt_fore[t] * infectious, 1e-9)
            
            # Генерируем случайное количество случаев
            y_t = nb_draw(mu_t, alpha_s)
            y_future_samples[s_idx, t] = y_t
            
            # Обновляем историю: сдвигаем окно и добавляем новый случай
            hist = np.concatenate([hist[1:], [y_t]])
    
    # Вычисляем статистики прогноза
    y_future_median = np.median(y_future_samples, axis=0)
    y_future_hdi = hdi(y_future_samples, prob=0.9)[..., :]
    
    # Создаем DataFrame с прогнозом
    future_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    fcst_df = pd.DataFrame({
        "date": future_dates,
        "pred_median": y_future_median,
        "pred_hdi_low": y_future_hdi[:, 0],
        "pred_hdi_high": y_future_hdi[:, 1]
    })
    
    # Получаем фактические данные для сравнения
    actual_future = full_window_df[
        (full_window_df["date"] >= forecast_start) & 
        (full_window_df["date"] <= forecast_end)
    ][["date", "new_cases"]].copy()
    
    # Объединяем прогноз с фактическими данными
    merged = pd.merge(fcst_df, actual_future, on="date", how="left")
    
    # Вычисляем метрики качества прогноза
    eval_df = merged.dropna(subset=["new_cases"]).copy()
    if not eval_df.empty:
        # Mean Absolute Error (MAE)
        ae = np.abs(eval_df["new_cases"] - eval_df["pred_median"])
        mae = float(np.nanmean(ae))
        
        # Mean Absolute Percentage Error (MAPE)
        ape = np.where(eval_df["new_cases"] > 0, ae / eval_df["new_cases"], np.nan)
        mape = float(np.nanmean(ape)) * 100.0
    else:
        mae = float("nan")
        mape = float("nan")
    
    results["metrics"] = {"MAE": mae, "MAPE_percent": mape}
    
    # =============================================================================
    # БЛОК СОХРАНЕНИЯ РЕЗУЛЬТАТОВ И ВИЗУАЛИЗАЦИИ
    # =============================================================================
    
    # Сохраняем сводку по R(t) (заглушка для совместимости)
    az_sum_path = os.path.join(outdir, f"{country}_Rt_summary.csv")
    pd.DataFrame({}).to_csv(az_sum_path, index=False)
    results["summary_files"]["Rt_summary_csv"] = az_sum_path
    
    # Сохраняем прогноз с фактическими данными
    fcst_path = os.path.join(outdir, f"{country}_forecast_vs_actual.csv")
    merged.to_csv(fcst_path, index=False)
    results["summary_files"]["forecast_csv"] = fcst_path
    
    # Вычисляем статистики R(t) для визуализации
    rt_median = np.median(rt_arr, axis=1)
    rt_hdi = hdi(rt_arr.T, prob=0.9)[..., :]
    
    # Создаем временные метки для R(t)
    train_dates = train_df["date"]
    all_dates = pd.date_range(train_dates.iloc[0], periods=lengths["T_total"], freq="D")
    
    # Сохраняем временной ряд R(t)
    rt_df = pd.DataFrame({
        "date": all_dates,
        "rt_median": rt_median,
        "rt_hdi_low": rt_hdi[:, 0],
        "rt_hdi_high": rt_hdi[:, 1]
    })
    rt_path = os.path.join(outdir, f"{country}_Rt_timeseries.csv")
    rt_df.to_csv(rt_path, index=False)
    results["summary_files"]["Rt_timeseries_csv"] = rt_path
    
    # Создаем график R(t)
    plt.figure(figsize=(12, 6))
    plt.plot(rt_df["date"], rt_df["rt_median"], linewidth=2, label="R(t) median")
    plt.fill_between(rt_df["date"], rt_df["rt_hdi_low"], rt_df["rt_hdi_high"], 
                     alpha=0.3, label="90% HDI")
    plt.axhline(1.0, color='red', linestyle="--", alpha=0.7, label="R(t) = 1")
    plt.title(f"{country} — Effective Reproduction Number R(t)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("R(t)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_rt_path = os.path.join(outdir, f"{country}_Rt.png")
    plt.savefig(fig_rt_path, dpi=150, bbox_inches='tight')
    plt.close()
    results["fig_files"]["Rt"] = fig_rt_path
    
    # Создаем график прогноза vs факт
    plt.figure(figsize=(12, 6))
    plt.plot(merged["date"], merged["pred_median"], linewidth=2, 
             label="Forecast median", color='blue')
    plt.fill_between(merged["date"], merged["pred_hdi_low"], merged["pred_hdi_high"], 
                     alpha=0.3, label="90% HDI", color='blue')
    
    if "new_cases" in merged.columns:
        plt.plot(merged["date"], merged["new_cases"], linewidth=2, 
                 label="Actual cases", color='red', marker='o', markersize=4)
    
    plt.title(f"{country} — Forecast vs Actual (daily new cases)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cases per day", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_fcst_path = os.path.join(outdir, f"{country}_Forecast_vs_Actual.png")
    plt.savefig(fig_fcst_path, dpi=150, bbox_inches='tight')
    plt.close()
    results["fig_files"]["Forecast_vs_Actual"] = fig_fcst_path
    
    print(f"[{country}] Done. MAE={mae:.1f}, MAPE={mape:.1f}%")
    return results

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ И ТОЧКА ВХОДА
# =============================================================================

def main(run_fit: bool = False) -> None:
    """
    Главная функция для анализа COVID-19 по всем странам.
    
    Эта функция координирует весь процесс анализа:
    1. Загружает данные COVID-19 из OWID
    2. Создает конфигурацию модели
    3. Обрабатывает каждую страну из списка COUNTRY_NAMES
    4. Сохраняет сводные метрики по всем странам
    5. Выводит информацию о результатах
    
    Args:
        run_fit (bool): Выполнять ли обучение моделей (по умолчанию False)
                       False - только построение моделей без обучения
                       True - полное обучение и прогнозирование
                       
    Note:
        Результаты сохраняются в директории 'outputs_rt_model/'
        Для каждой страны создается поддиректория с результатами
    """
    # Создаем основную выходную директорию
    outdir = "outputs_rt_model"
    ensure_dir(outdir)
    
    # Загружаем данные COVID-19
    print("Loading COVID-19 data from OWID...")
    df = load_owid_data()
    
    # Создаем конфигурацию модели
    cfg = ModelConfig()
    
    # Словарь для хранения результатов по всем странам
    all_results = {}
    
    # Обрабатываем каждую страну
    for country in COUNTRY_NAMES.values():
        print("=" * 80)
        print(f"Processing {country}")
        print("=" * 80)
        
        try:
            # Создаем поддиректорию для страны
            country_outdir = os.path.join(outdir, country.replace(' ', '_'))
            
            # Выполняем анализ для страны
            res = fit_and_forecast_country(
                country=country,
                full_df=df,
                outdir=country_outdir,
                config=cfg,
                run_fit=run_fit
            )
            all_results[country] = res
            
        except Exception as e:
            print(f"[{country}] ERROR: {e}")
            all_results[country] = {"error": str(e)}
    
    # Создаем сводную таблицу метрик
    metrics_rows = []
    for country, result in all_results.items():
        row = {"country": country}
        row.update(result.get("metrics", {"MAE": np.nan, "MAPE_percent": np.nan}))
        metrics_rows.append(row)
    
    # Сохраняем сводные метрики
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(outdir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    print("[summary] Metrics written to:", metrics_path)
    print("\nAll done. Artifacts saved under:", outdir)
    
    if not run_fit:
        print("Tip: set run_fit=True in main() to actually run sampling.")
    else:
        print("Full analysis completed with MCMC sampling.")


main(run_fit=True)
