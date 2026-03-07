import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyparsing import results


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """

    return np.arange(0, 10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """

    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5)

    Args:
        vec (numpy.ndarray): Входной массив формы (10,)

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """

    # return np.reshape(vec, (2, 5))
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """

    return np.transpose(mat)


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """

    return a + b


def scalar_multiply(vec: np.ndarray, scalar: int | float) -> np.ndarray:
    """
    Умножение вектора на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """

    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """

    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> int | float:
    """
    Скалярное произведение.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов
    """

    return np.dot(a, b)

# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц
    """

    return np.matmul(a, b)


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы
    """

    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица
    """

    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x
    """

    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================


def load_dataset(path: str ="data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """

    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict:
    """
    Представьте, что данные — это результаты экзамена по математике.
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Args:
        data (numpy.ndarray): Одномерный массив данных

    Returns:
        dict: Словарь со статистическими показателями
    """

    stats = {}

    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)

    stats.update({"mean": mean, "median": median, "std": std, "min": minimum, "max": maximum,
                    "percentile_25": percentile_25, "percentile_75": percentile_75})

    return stats


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация.

    Формула: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """

    minimum = np.min(data)
    maximum = np.max(data)

    normalized_data = (data - minimum) / (maximum - minimum)

    return normalized_data


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data (numpy.ndarray): Данные для гистограммы
    """

    plt.hist(data, bins=50, label="математика")
    plt.title("Распределение оценок по математике")
    plt.xlabel("Балл")
    plt.ylabel("Количество работ")
    plt.legend()
    plt.savefig("plots/histogram.png")
    plt.show()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции предметов.

    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """

    sns.heatmap(matrix, annot=True)
    plt.title("Тепловая карта корреляции предметов")
    plt.savefig("plots/heatmap.png")
    plt.show()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.

    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """

    plt.plot(x, y)
    plt.title("график зависимости оценки студента от его номера")
    plt.xlabel("Номера студентов")
    plt.ylabel("Оценки студентов")
    plt.savefig("plots/line.png")
    plt.show()
