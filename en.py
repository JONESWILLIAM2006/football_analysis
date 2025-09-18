import os

src = input("Source image path: ")                 # full path to .png
dst = input("Destination filename: ")               # e.g. copy.png

if not os.path.exists(src):
    raise FileNotFoundError(f"{src} not found")

with open(src, "rb") as fi, open(dst, "wb") as fo:
    while chunk := fi.read(8192):                  # read in 8K blocks
        fo.write(chunk)

print("Copied", src, "→", dst)