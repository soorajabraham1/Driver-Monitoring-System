import cv2
import numpy as np

# Function to draw a filled rounded rectangle with transparency
def draw_filled_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=0.5):
    # Create an overlay for the rounded rectangle
    overlay = frame.copy()

    # Define the corner points
    top_left_radius = (top_left[0] + radius, top_left[1] + radius)
    top_right_radius = (bottom_right[0] - radius, top_left[1] + radius)
    bottom_left_radius = (top_left[0] + radius, bottom_right[1] - radius)
    bottom_right_radius = (bottom_right[0] - radius, bottom_right[1] - radius)

    # Draw the four corners
    cv2.ellipse(overlay, top_left_radius, (radius, radius), 180, 0, 90, color, cv2.FILLED)
    cv2.ellipse(overlay, top_right_radius, (radius, radius), 270, 0, 90, color, cv2.FILLED)
    cv2.ellipse(overlay, bottom_left_radius, (radius, radius), 90, 0, 90, color, cv2.FILLED)
    cv2.ellipse(overlay, bottom_right_radius, (radius, radius), 0, 0, 90, color, cv2.FILLED)

    # Draw the four edges
    cv2.rectangle(overlay, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), color, cv2.FILLED)
    cv2.rectangle(overlay, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, cv2.FILLED)

    # Apply the overlay with alpha blending
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame

# # Load the frame
# frame = cv2.imread('users\Sooraj.jpg')  # Replace with your frame capture logic

# # Define the rectangle properties
# top_left = (50, 50)
# bottom_right = (300, 150)
# color = (0, 255, 0)  # Green color
# radius = 20
# alpha = 0.5  # Transparency factor

# # Draw the filled rounded rectangle with transparency
# frame = draw_filled_rounded_rectangle(frame, top_left, bottom_right, color, radius, alpha)

# # Display the result
# cv2.imshow('Frame with Filled Rounded Rectangle', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
