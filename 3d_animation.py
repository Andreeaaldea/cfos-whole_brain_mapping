import napari
import numpy as np
import imageio

# Assume the viewer and your image layer have been setup as shown above.
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data, name='Large Volume')

# Define number of frames and initialize a list for frames
num_frames = 60  # e.g., 60 frames for a full rotation
frames = []

# Define a function to update the view (e.g., rotating around the Y-axis)
for i in range(num_frames):
    # Calculate the rotation angle in degrees
    angle = (360 / num_frames) * i
    # Update the camera; adjust the axis based on your dataset and desired rotation
    viewer.camera.angles = (0, angle, 0)  # (x, y, z) rotation in degrees

    # Optionally, you might want to force a render update:
    viewer.reset_view()
    napari.utils.notifications.show_info(f"Capturing frame {i+1}/{num_frames}")
    
    # Capture screenshot (set canvas_only=True to capture the rendered canvas)
    # Pause briefly if necessary to ensure the rendering is updated.
    screenshot = viewer.screenshot(canvas_only=True)
    frames.append(screenshot)

# Save frames as a video using imageio or a similar library
writer = imageio.get_writer('napari_animation.mp4', fps=20)
for frame in frames:
    writer.append_data(frame)
writer.close()
