import cv2
from PIL import Image
method = cv2.TM_SQDIFF_NORMED
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_frog_position(isize,im):
	frog =  Image.open('./png/frog.png')
	fsize = frog.size
	x0, y0 = fsize [0] // 2, fsize [1] // 2
	pixel = frog.getpixel((x0 + 10, y0 + 10))[:-1]
	best = (100000, 0, 0)
	for x in range (isize[0]):
		for y in range (isize[1]):
			ipixel = im.getpixel ((x, y))
			d = diff (ipixel, pixel)
			if d < best[0]: best = (d, x, y)
	x, y = best [1:]
	return x,y,x0,y0

def diff (a, b):
    return sum ( (a - b) ** 2 for a, b in zip(a, b) )

def get_car_position(isize,im):
	frog =  Image.open('./png/redcar.png')
	fsize = frog.size
	x0, y0 = fsize [0] // 2, fsize [1] // 2
	pixel = frog.getpixel((x0 + 10, y0 + 10))[:-1]
	best = (100000, 0, 0)
	for x in range (isize[0]):
		for y in range (isize[1]):
			ipixel = im.getpixel ((x, y))
			d = diff (ipixel, pixel)
			if d < best[0]: best = (d, x, y)
	x, y = best [1:]
	return x,y,x0,y0

def matchTemplate(searchImage, templateImage):
    minScore = -1000
    searchWidth = searchImage.size[0]
    searchHeight = searchImage.size[1]
    templateWidth = templateImage.size[0]
    templateHeight = templateImage.size[1]
    #loop over each pixel in the search image
    for xs in range(searchWidth):
        for ys in range(searchHeight):
            #set some kind of score variable to 0
            score = 0
            #loop over each pixel in the template image
            for xt in range(templateWidth):
                for yt in range(templateHeight):
                    if xs+xt < 0 or ys+yt < 0 or xs+xt >= searchWidth or ys+yt >= searchHeight:
                        score += 0
                    else:
                        pixel_searchImage = searchImage.getpixel(((xs+xt),(ys+yt)))
                        pixel_templateImage = templateImage.getpixel((xt, yt))
                        if pixel_searchImage == pixel_templateImage:
                            score += 1
                        else: score -= 1
            if minScore < score:
                minScore = score
                matching_xs = xs
                matching_ys = ys
                matching_xt = xt
                matching_yt = yt
                matching_score = score
                
    print matching_xs, matching_ys, matching_xt, matching_yt, matching_score
    im1 = Image.new('RGB', (searchWidth, searchHeight), (80, 147, 0))
    im1.paste(templateImage, ((matching_xs), (matching_ys)))
    searchImage.show()
    im1.show()
    im1.save('template_matched_in_search.jpg')


def match_image():
	img = cv2.imread('./png//Frogger_State_1.png',0)
	img2 = img.copy()
	template = cv2.imread('./png/frog.png',0)
	w, h = template.shape[::-1]
	w = w
	h = h + 10

	# All the 6 methods for comparison in a list
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
	            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	# for meth in methods:
	meth = methods[1]
	img = img2.copy()
	method = eval(meth)

	# Apply template Matching
	res = cv2.matchTemplate(img,template,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
	    top_left = min_loc
	else:
	    top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(img,top_left, bottom_right, 255, 2)

	plt.subplot(121),plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.suptitle(meth)

	plt.show()
# searchImage = Image.open("./png/Screenshot_0.png")
# templateImage = Image.open("./png/frog.png")
# matchTemplate(searchImage, templateImage)
# print "end"

def multiple_object_match():
	img_rgb = cv2.imread('./png/Frogger_State_1.png')
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('./png/frog.png',0)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.4
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
	    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

	cv2.imwrite('res.png',img_rgb)

# multiple_object_match()
# match_image()

def parse_image(s_image,l_image):
	small_image = cv2.imread(s_image)
	large_image = cv2.imread(l_image)

	result = cv2.matchTemplate(small_image, large_image, method)

	# We want the minimum squared difference
	mn,_,mnLoc,_ = cv2.minMaxLoc(result)

	# Draw the rectangle:
	# Extract the coordinates of our best match
	MPx,MPy = mnLoc

	# Step 2: Get the size of the template. This is the same size as the match.
	trows,tcols = small_image.shape[:2]

	# Step 3: Draw the rectangle on large_image
	cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

	# Display the original image with the rectangle around the match.
	cv2.imshow('output',large_image)
	cv2.imwrite('res.png',large_image)

	# # The image is only displayed if we call this
	# cv2.waitKey(0)

# parse_image('./png/redcar.png','./png/Frogger_State_1.png')
im = Image.open('./png/Screenshot_2.png')
isize = im.size
# next_isize = next_im.size
x, y, x0, y0 = get_car_position(isize,im)
print(x,y)