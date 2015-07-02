import os

def apply_all_images(input_folder, function, output_folder=None):
	images = [image_file
			  for image_file
			  in os.listdir(input_folder)
			  if os.path.splitext(image_file)[1] == '.JPG']

	for image_file in images:
		if output_folder is not None:
			function(os.path.join(input_folder, image_file), output_folder)
		else:
			function(os.path.join(input_folder, image_file))