import pcl
import cv2
import numpy as np
import math


def color_by_intensity(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try:
        x = float(mag - cmin) / (cmax - cmin)
    except ZeroDivisionError:
        x = 0.5  # cmax == cmin
    blue = min((max((4 * (0.75 - x), 0.)), 1.))
    red = min((max((4 * (x - 0.25), 0.)), 1.))
    green = min((max((4 * math.fabs(x - 0.5) - 1., 0.)), 1.))
    return int(red * 255), int(green * 255), int(blue * 255)


def load_pcd(pcd_path):
    pcd_map = pcl.load_XYZI(pcd_path)
    w = pcd_map.width
    h = pcd_map.height
    num_points = pcd_map.size
    print("Map width: " + str(w) + " , height: " + str(h))
    print("Number of points: " + str(num_points))
    return pcd_map


def generate_bev(pcd_map, resolution):
    min_x, min_y, max_x, max_y = 0, 0, 0, 0
    min_i, max_i = 0, 0

    # Extract dimensions of area spanned by point cloud
    for i in range(pcd_map.size):
        if pcd_map[i][0] < min_x:
            min_x = pcd_map[i][0]
        if pcd_map[i][0] > max_x:
            max_x = pcd_map[i][0]
        if pcd_map[i][1] < min_y:
            min_y = pcd_map[i][1]
        if pcd_map[i][1] > max_y:
            max_y = pcd_map[i][1]
        if pcd_map[i][3] < min_i:
            min_i = pcd_map[i][3]
        if pcd_map[i][3] > max_i:
            max_i = pcd_map[i][3]

    # Total area in map frame
    total_x = abs(max_x - min_x)
    total_y = abs(max_y - min_y)

    # Total area in image frame
    pix_x = int(total_x / resolution)
    pix_y = int(total_y / resolution)

    # Dimensions of BEV image
    bev_img = np.zeros((pix_x + 1, pix_y + 1, 3))

    # Identify origin tf
    x_origin = abs(min_x)
    y_origin = abs(min_y)

    print("Pixel origin: " + str(x_origin) + " , " + str(y_origin))
    print("Resolution: " + str(resolution))

    # Convert 3d point to image point
    for i in range(pcd_map.size):
        u_i = int((pcd_map[i][0] + x_origin) / resolution)
        v_i = int((pcd_map[i][1] + y_origin) / resolution)
        r, g, b = color_by_intensity(pcd_map[i][3], 0.0, 255.0)
        bev_img[u_i, v_i, 0] = g
        bev_img[u_i, v_i, 1] = r
        bev_img[u_i, v_i, 2] = b

    cv2.imwrite('bev.jpg', bev_img)
    # cv2.imshow('image', bev_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # map = load_pcd("/home/dfpazr/Downloads/table_scene_lms400.pcd")
    map = load_pcd("/home/dfpazr/Documents/CogRob/avl/Maps/MailRoute/mail-route.pcd")
    img = generate_bev(map, resolution=0.05)
