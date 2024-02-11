import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def find_points(contours):
    points = []
    for i in contours:
        (x, y), radius = cv2.minEnclosingCircle(i)
        points.append([x, y])
    return points


def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    img_for_print = copy.deepcopy(img)
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img_for_print, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img_for_print, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img_for_print, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    plt.imshow(img_for_print)
    plt.show()
    return triangleList


def coordinates(triangle_list, blue_points, red_points_sorted):
    coordinates_blue = []
    coordinates_red = []
    for t in triangle_list:
        cord_1 = [t[0], t[1]]
        cord_2 = [t[2], t[3]]
        cord_3 = [t[4], t[5]]
        cord_1_red = red_points_sorted[blue_points.index(cord_1)]
        cord_2_red = red_points_sorted[blue_points.index(cord_2)]
        cord_3_red = red_points_sorted[blue_points.index(cord_3)]
        coordinates_red.append([cord_1_red, cord_2_red, cord_3_red])
        coordinates_blue.append([cord_1, cord_2, cord_3])
    return coordinates_blue, coordinates_red


def warpTriangle(img1, img2, tri1, tri2):
    # Находим границы для каждого треугольника по набору координат его вершин
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    tri1Cropped = []
    tri2Cropped = []

    for i in range(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

    # Обрезаем изображение по треугольнику
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Находим аффинное преобразование по двум соответствующим треугольникам
    warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

    # Применяем найденную матрицу к выделенному треугольнику
    img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Убираем пиксели, выходящие за границу области, для последующего наложения
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

    img2Cropped = img2Cropped * mask

    # Вставляем преобразованную область в соответствующую область на выходном изображении
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Cropped


if __name__ == '__main__':

    # Считывание фото
    blue_image = cv2.imread('templates/example_of_pp/pp_blue_with_print_1.png')
    red_image = cv2.imread('templates/example_of_pp/pp_red.png')

    # Перевод в цветовое пространство HSV
    hsv_blue_image = cv2.cvtColor(blue_image, cv2.COLOR_BGR2HSV)
    hsv_red_image = cv2.cvtColor(red_image, cv2.COLOR_BGR2HSV)

    # Вывод преобразованных изображений на графиках
    plt.title('Blue')
    plt.imshow(hsv_blue_image)
    plt.show()
    plt.title('red')
    plt.imshow(hsv_red_image)
    plt.show()

    # Применение маски для нахождения кругов отверстий
    circles_blue = cv2.inRange(hsv_blue_image, np.array([119, 255, 255]), np.array([121, 255, 255]))
    circles_red = cv2.inRange(hsv_red_image, np.array([0, 10, 0]), np.array([125, 255, 255]))
    contours_circles_blue, _ = cv2.findContours(circles_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_circles_red, _ = cv2.findContours(circles_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Находим координаты красных и синих отверстий
    blue_points = find_points(contours_circles_blue)
    red_points = find_points(contours_circles_red)

    # Сортируем найденные точки в списке по координате Y;
    # Красные точки записываются в соответствии с синими по минимальному расстоянию между ними
    blue_points = sorted(blue_points, key=lambda x:x[1])
    red_points = sorted(red_points, key=lambda x:x[1])
    red_points_sorted = []
    for _, b in enumerate(blue_points.copy()):
        arr = []
        for i, r in enumerate(red_points):
            arr.append(((b[0] - r[0]) ** 2 + (b[1] - r[1]) ** 2)**0.5)
        red_points_sorted.append(red_points[arr.index(min(arr))])

    # Вывод координат отверстий в консоль
    print(f'Points:\n')
    for i in range(len(blue_points)):
        print(f'{i}:  {blue_points[i]}         {red_points_sorted[i]}')

    size = cv2.cvtColor(blue_image, cv2.COLOR_HSV2BGR).shape
    rect = (0, 0, size[1], size[0])
    subdiv_blue = cv2.Subdiv2D(rect)
    subdiv_red = cv2.Subdiv2D(rect)
    for p in copy.deepcopy(blue_points):
        subdiv_blue.insert(p)
        p[0] = int(p[0])
        p[1] = int(p[1])
        draw_point(cv2.cvtColor(blue_image, cv2.COLOR_HSV2BGR), tuple(p), (0, 0, 255))

    for p in red_points_sorted.copy():
        subdiv_red.insert(p)
        p[0] = int(p[0])
        p[1] = int(p[1])
        draw_point(cv2.cvtColor(red_image, cv2.COLOR_HSV2BGR), tuple(p), (0, 0, 255))

    # Отрисовка найденных областей (треугольников) и получение списка координат их вершин
    triangleList_blue = draw_delaunay(blue_image, subdiv_blue, (255, 255, 255))
    triangleList_red = draw_delaunay(red_image, subdiv_red, (255, 255, 255))

    # Составление списка координат для последующего преобразования
    triangleCords_blue, triangleCords_red = coordinates(triangleList_blue, blue_points, red_points_sorted)

    # Вывод координат вершин треугольных областей
    print(f'\n\ntriangleLists:\n')
    for i in range(len(triangleCords_blue)):
        print(f'{i}:     {triangleCords_blue[i]}          {triangleCords_red[i]}')

    # Преобразование областей по элементам из списка координат вершин треугольных областей
    for i, _ in enumerate(triangleCords_blue):
        warpTriangle(blue_image, red_image, np.float32([triangleCords_blue[i]]), np.float32([triangleCords_red[i]]))

    # Вывод и сохранение выходного преобразованного изображения
    cv2.imwrite('output/output.png', red_image)
    plt.title('Outcome image')
    plt.imshow(red_image)
    plt.show()
    cv2.waitKey(0)
