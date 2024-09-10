import cv2
import numpy as np

def find_keypoints_and_matches(img1, img2):
    # Инициализация детектора SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Поиск ключевых точек и дескрипторов
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Используем FLANN-матчер для поиска соответствий
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Применяем тест Лоу для фильтрации плохих соответствий
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def triangulate_points(img1, img2, keypoints1, keypoints2, matches, baseline, focal_length):
    # Выбор точек соответствия
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Преобразование координат в однородные
    points1_hom = cv2.convertPointsToHomogeneous(points1)
    points2_hom = cv2.convertPointsToHomogeneous(points2)

    # Матрицы камеры для левого и правого изображений
    P1 = np.array([[focal_length, 0, img1.shape[1] / 2],
                   [0, focal_length, img1.shape[0] / 2],
                   [0, 0, 1]])
    P2 = np.array([[focal_length, 0, img2.shape[1] / 2],
                   [0, focal_length, img2.shape[0] / 2],
                   [0, 0, 1]])

    # Расстояние между камерами (базис)
    T = np.array([baseline, 0, 0])

    # Триангуляция
    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

    # Преобразуем из однородных координат в 3D
    points_3d = points_4d / points_4d[3]

    return points_3d[:3].T

def calculate_focal_length_in_pixels(f_mm, sensor_width_mm, image_width_pixels):
    """
    Рассчитать фокусное расстояние в пикселях
    :param f_mm: Фокусное расстояние в миллиметрах
    :param sensor_width_mm: Ширина сенсора камеры в миллиметрах
    :param image_width_pixels: Ширина изображения в пикселях
    :return: Фокусное расстояние в пикселях
    """
    return (f_mm * image_width_pixels) / sensor_width_mm


if __name__ == "__main__":
    # Загрузка изображений
    img1 = cv2.imread('test1001.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('test1002.png', cv2.IMREAD_GRAYSCALE)

    # Параметры камеры
    f_mm = 80.0  # фокусное расстояние камеры в миллиметрах
    sensor_width_mm = 63.0  # ширина сенсора
    image_width_pixels = 1920  # ширина изображения в пикселях
    baseline = 0.412  # расстояние между камерами (в метрах)
    focal_length = calculate_focal_length_in_pixels(f_mm, sensor_width_mm, image_width_pixels)  # фокусное расстояние (в пикселях)

    # Находим ключевые точки и соответствия
    keypoints1, keypoints2, good_matches = find_keypoints_and_matches(img1, img2)

    # Триангуляция и получение 3D-точек
    points_3d = triangulate_points(img1, img2, keypoints1, keypoints2, good_matches, baseline, focal_length)

    print("3D Points:\n", points_3d)
