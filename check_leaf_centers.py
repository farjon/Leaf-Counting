import os
import csv
from GetEnvVar import GetEnvVar
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import copy

def read_imgs_data(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1


        img_file, x, y = row[:3]

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x, y) == ('', ''):
            raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

        result[img_file].append({'x': x, 'y': y})
    return result


def main(Params):
    for set in Params['sets']:
        data_path = os.path.join(Params['data_main_path'], set)
        annotations_file_name = os.path.join(data_path, 'BL_' + set.title() +'_leaf_location.csv')
        with open(annotations_file_name, 'r', newline='') as file:
            image_data_leaf_location = read_imgs_data(csv.reader(file, delimiter=','))
        for image_data in image_data_leaf_location:
            im_original = Image.open(os.path.join(data_path, image_data))
            im_drawing = copy.deepcopy(im_original)
            draw_mask = ImageDraw.Draw(im_drawing)
            for point in image_data_leaf_location[image_data]:
                x_tl = round(int(point['x']) - 6)
                x_br = round(int(point['x']) + 6)
                y_tl = round(int(point['y']) - 6)
                y_br = round(int(point['y']) + 6)
                draw_mask.ellipse([x_tl, y_tl, x_br, y_br], fill='red')
            # im_drawing.show()
            new_file_name = os.path.join(Params['drawing_path'], image_data)
            im_drawing.save(new_file_name)

if __name__ == "__main__":
    Params = {}
    Params['data_main_path'] = os.path.join(GetEnvVar("DatasetsPath"),  "Counting Datasets", "Banana_leaves", "BL")
    Params['sets'] = ['train', 'test', 'val']
    Params['drawing_path'] = os.path.join(Params['data_main_path'], "leaf_center_drawing")
    if not os.path.exists(Params['drawing_path']):
        os.makedirs(Params['drawing_path'])
    main(Params)