import os
import sys

def apply_all_images(input_folder, function, output_folder=None):
    images = [image_file
              for image_file
              in os.listdir(input_folder)
              if os.path.splitext(image_file)[1].lower() == '.jpg']

    # Ignoring exceptions only for the sake of not interrupting during batch processing
    for image_file in images:
        if output_folder is not None:
            try:
                function(os.path.join(input_folder, image_file), output_folder)
            except Exception:
                sys.exc_clear()
        else:
            try:
                function(os.path.join(input_folder, image_file))
            except Exception:
                sys.exc_clear()

def specimen_ids_from_images(filenames, prefix='color_'):
    prefix_len = len(prefix)
    for filename in filenames:
        if filename.startswith(prefix):
            yield filename[prefix_len:]

def get_specimen_ids(filename):
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)

        specimen_ids = []

        for row in reader:
            sex = row[3]
            if sex == 'male' or sex == 'female':
                ids = glob.glob(os.path.join('data','full_image',sex,'*{}*'.format(row[0])))
                if ids:
                    if len(ids) > 1:
                        ids = sorted(ids, key=len)
                    specimen_ids.append((row[0], ids[0]))

    return specimen_ids