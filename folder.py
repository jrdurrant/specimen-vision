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