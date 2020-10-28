from collections import deque 
import numpy as np

#checkign for visibility
VISIBLE_THRESHOLD = np.linalg.norm(np.array([20, 20, 20]))


#performs floodfill cuz cv2.floodfill is bad
#sets flooded to elements to value
def flood_fill_bit_vis(seed_loc, image, value, thresh = 0.02):

	mask = np.zeros((image.shape[0], image.shape[1]))
	mask_visible = np.zeros((image.shape[0], image.shape[1]))

	q = deque()
	q.append(seed_loc)
	while len(q) > 0:
		loc = q.popleft()

		#already checked this place
		if mask[loc[1]][loc[0]] == value:
			continue
		
		current_val = image[loc[1]][loc[0]]
		current_val_norm = np.linalg.norm(current_val)

		mask[loc[1]][loc[0]] = value

		#invisible, nothing to floodfill
		if current_val_norm < VISIBLE_THRESHOLD:
			continue

		mask_visible[loc[1]][loc[0]] = value

		for i in range(-1, 2):
			for k in range(-1, 2):

				if loc[1] + i < 0 or loc[1] + i >= image.shape[0] or loc[0] + k < 0 or loc[0] + k >= image.shape[1]:
					continue

				val = image[loc[1] + i][loc[0] + k]
				if 1 - np.linalg.norm(val) / current_val_norm < thresh:
					q.append((loc[0] + k, loc[1] + i))

	return mask, mask_visible



