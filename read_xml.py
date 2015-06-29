import xml.etree.ElementTree as ET
import numpy as np
import re
import tempfile

def set_hog_xml_parameters(filename, win_size):
	tree = ET.parse(filename)

	win_size_element = tree.getroot().getchildren()[0].find('winSize')
	text = win_size_element.text
	match = re.match('([^0-9]*)([1-9][0-9]*)([^0-9]*)([1-9][0-9]*)([^0-9]*)', text)
	output = []
	for m in match.groups():
		output.append(m)
	output[1] = win_size[0]
	output[3] = win_size[1]
	new_text = ''
	for string in output:
		new_text += str(string)
	win_size_element.text = new_text

	f = open(filename, 'w')
	f.write('<?xml version="1.0"?>\n' + ET.tostring(tree.getroot()))
	f.close()

def get_support_vector_from_xml(filename):
	tree = ET.parse(filename)

	sv = []
	for s in tree.iter('support_vectors'):
		sv.append(s)

	text = sv[0].getchildren()[0].text
	text2 = text.strip().split('\n')
	text3 = []
	for string in text2:
		new_string = string.split()
		text3.append(new_string)

	rows = len(text3)
	cols = len(text3[0])
	support_vector = np.zeros((rows * cols, 1), dtype='float32')
	for i in range(0, rows):
		for j in range(0, cols):
			support_vector[i*cols + j] = float(text3[i][j])

	for s in tree.iter('rho'):
		rho = float(s.text)

	return support_vector, rho