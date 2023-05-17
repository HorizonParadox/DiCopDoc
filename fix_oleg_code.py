import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import doc_rec_cls as DC
from image_crop import image_crop

NUM_OF_LINE = 12
POLYNOMIAL_DEGREE = 4  # curve polynomial degree, better 3 or 4
INPUT_DIRECTORY = "dataset"
INPUT_DIRECTORY_DEPTH = "document_depth_maps"
OUTPUT_DIRECTORY = "output"

for filename1 in os.listdir(INPUT_DIRECTORY):
    file = os.path.join(INPUT_DIRECTORY, filename1)

    if os.path.isfile(file):
        print(filename1)
        initial_img = cv2.imread(file)
        #initial_img = cv2.resize(initial_img, (initial_img.shape[1] // 2, initial_img.shape[0] // 2))
        initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB)
        height_image, width_image, _ = initial_img.shape


        num_points = int(width_image * 0.1)
        print(f'num_points: {num_points}')
        num_of_mask = 0
        # Функция Get mask и поиск контура
        polynomial = DC.draw_polynomial(initial_img.copy(), thres_coef=0.03)
        for concave, conv, cont in polynomial:
            num_of_mask += 1

            # Построение контура
            DC.plot_polynomial(initial_img.copy(), concave, conv, cont)

            # Поиск сторон в контуре
            left, right, top, bottom = DC.find_edges(initial_img.copy(), cont, conv)

            # Получение приблизительной длины и ширины контура
            length, width = DC.get_dimensions(left, right, top, bottom)
            print(f'LW: {length, width, left[0]}')

            # Отображение полученых сторон разным цветом
            DC.plot_colorful_edges(initial_img.copy(), left, right, top, bottom)

            # doc_rec_cls.zip_coords(left, right, bottom, top)
            # Интерполяция двух кривых
            new_left_x, new_left_y, new_right_x, new_right_y = DC.opposite_interpolate(left, right, True)
            new_bottom_x, new_bottom_y, new_top_x, new_top_y = DC.opposite_interpolate(bottom, top, True)

            # doc_rec_cls.useless_info(new_top_x, new_top_y, new_bottom_x, new_bottom_y)

            # Построение линий аппроксимации (сетка)
            linfit_x_bt, linfit_y_bt, linfit_x_lr, linfit_y_lr = DC.get_lines_of_approximation(
                initial_img.copy(), NUM_OF_LINE, new_left_x, new_left_y, new_right_x, new_right_y, new_bottom_x,
                new_bottom_y, new_top_x, new_top_y)

            plt.imshow(initial_img)
            # Построение интерполяции аппроксимации (горизонт, вертикал)
            # !Подумать о другом способе вычисления z
            polys_bt = DC.interpolation_approximation_3(
                initial_img.copy(), width_image, NUM_OF_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_bt, linfit_y_bt)
            polys_lr = DC.interpolation_approximation_3(
                initial_img.copy(), height_image, NUM_OF_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_lr, linfit_y_lr)
            plt.show()

            plt.imshow(initial_img)
            #  Поиск пересечений между линиями, заданными интерполяционными аппроксимациями
            points_set = np.asarray(DC.get_intersection_interpolation_approximation(length, width, polys_bt, polys_lr))
            plt.show()

            shift = left[0]
            # Получение координат исправленной сетки
            XYpairs = DC.get_fix_grid_coordinates(length, width, NUM_OF_LINE, shift)
            plt.imshow(initial_img)
            plt.plot(XYpairs[:, 0], XYpairs[:, 1], marker='.', color='k', linestyle='none')
            plt.show()

            # Восстановление искажения перспективы изображения
            warped_image = DC.remap_image(initial_img.copy(), height_image, width_image, points_set, XYpairs)
            crop_warped_image = image_crop(warped_image)

            # Превращает документ в чёрно-белый скан
            # scan = DC.get_black_white_scan(crop_warped_image)
            crop_warped_image_BGR = cv2.cvtColor(crop_warped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(OUTPUT_DIRECTORY + f'/{num_of_mask}__kv' + filename1, crop_warped_image_BGR)

            plt.subplot(121, title='before')
            plt.imshow(initial_img)
            plt.axis('off')
            plt.subplot(122, title='after')
            plt.imshow(crop_warped_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
