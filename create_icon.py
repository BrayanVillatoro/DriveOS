"""
Create a DriveOS icon file with checkered flag design
"""
from PIL import Image, ImageDraw

# Create 256x256 icon
size = 256
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Colors
black = (30, 30, 30, 255)
white = (255, 255, 255, 255)
blue = (0, 122, 204, 255)  # VS Code accent blue

# Checkered flag pattern (8x8 squares)
square_size = size // 8
for row in range(8):
    for col in range(8):
        # Checkered pattern
        if (row + col) % 2 == 0:
            color = black
        else:
            color = white
        
        x = col * square_size
        y = row * square_size
        draw.rectangle([x, y, x + square_size, y + square_size], fill=color)

# Add blue border/frame
border = 8
draw.rectangle([0, 0, size, border], fill=blue)  # Top
draw.rectangle([0, size-border, size, size], fill=blue)  # Bottom
draw.rectangle([0, 0, border, size], fill=blue)  # Left
draw.rectangle([size-border, 0, size, size], fill=blue)  # Right

# Save as ICO with multiple sizes
img.save('DriveOS.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("Icon created: DriveOS.ico")
