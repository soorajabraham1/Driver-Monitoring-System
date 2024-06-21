import cv2
import numpy as np
def myUi(frame, overlay_image):
    # Load the frame (or capture from a video source)
    #frame = cv2.imread('../icons/black.jpg')  # Replace with your frame capture logic
    #frame = cv2.resize(frame, (640, 480))
    # Load the image to overlay (e.g., a logo or icon)
    #overlay_image = cv2.imread('../icons/profile.jpg', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    # Resize the overlay image if needed
    #overlay_image = cv2.resize(overlay_image, (50, 50))  # Resize to desired size

    # Extract the alpha channel from the overlay image
    if overlay_image.shape[2] == 4:  # Check if the overlay has an alpha channel
        alpha_channel = overlay_image[:, :, 3]
        overlay_image = overlay_image[:, :, :3]
    else:
        alpha_channel = np.ones(overlay_image.shape[:2], dtype=np.uint8) * 255

    # Define the location to place the overlay image on the frame
    x_offset = 10
    y_offset = 30
    x_end = x_offset + overlay_image.shape[1]
    y_end = y_offset + overlay_image.shape[0]

    # Define the region of interest (ROI) on the frame
    roi = frame[y_offset:y_end, x_offset:x_end]

    # Varying transparency: adjust alpha value
    transparency = 1  # Set transparency level (0.0 to 1.0)
    alpha_channel = (alpha_channel * transparency).astype(np.uint8)

    # Blend the overlay image with the ROI
    alpha = alpha_channel / 255.0
    for c in range(0, 3):
        roi[:, :, c] = (alpha * overlay_image[:, :, c] + (1 - alpha) * roi[:, :, c])
    

    #Put the ROI back into the frame
    frame[y_offset:y_end, x_offset:x_end] = roi
    return frame
# # Display the result
# cv2.imshow('Frame with Overlay', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
