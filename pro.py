import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import os
import shutil

# Download a real pitch image if not present
import urllib.request
pitch_path = "football_pitch.jpg"
if not os.path.exists(pitch_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opendatasets/football-pitch/main/pitch.jpg",  # ✅ Working Image
        pitch_path
    )

# Explicitly find and set the ffmpeg path for Matplotlib
ffmpeg_executable = shutil.which('ffmpeg')
if ffmpeg_executable:
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_executable
else:
    print("WARNING: ffmpeg not found. Animation saving will likely fail.")
    print("Please install ffmpeg and ensure it is in your system's PATH.")

# Load pitch image
pitch = plt.imread(pitch_path)

# Simulated player positions (x, y in pixel coordinates)
team_a = {
    7: (200, 300),  # player who made the wrong pass
    8: (400, 280),  # marked
    9: (600, 350),  # unmarked
    10: (750, 250), # unmarked
}
team_b = {
    4: (420, 290),  # near 8
    5: (550, 300),  # near 9
    6: (800, 300),  # far from 10
}

wrong_pass_from = 7
wrong_pass_to = 8

# Tactical suggestion logic
def suggest_better_pass(from_id, team_a, team_b):
    from_pos = np.array(team_a[from_id])
    best_target = None
    max_distance = 0

    for pid, pos in team_a.items():
        if pid == from_id or pid == wrong_pass_to:
            continue

        opponent_dist = min(np.linalg.norm(np.array(pos) - np.array(def_pos)) for def_pos in team_b.values())
        if opponent_dist > 100:  # safe threshold
            distance = np.linalg.norm(np.array(pos) - from_pos)
            if distance > max_distance:
                max_distance = distance
                best_target = (pid, pos)

    return best_target

suggested_id, suggested_pos = suggest_better_pass(wrong_pass_from, team_a, team_b)

# Setup animation
fig, ax = plt.subplots()
ax.set_xlim(0, pitch.shape[1])
ax.set_ylim(pitch.shape[0], 0)
ax.imshow(pitch)

dots = {}
texts = []

for pid, pos in team_a.items():
    dot, = ax.plot([], [], 'bo', markersize=12)
    dots[pid] = dot
    texts.append(ax.text(pos[0], pos[1] - 15, f"A{pid}", color='blue', fontsize=10))

for pid, pos in team_b.items():
    dot, = ax.plot([], [], 'ro', markersize=12)
    dots[pid] = dot
    texts.append(ax.text(pos[0], pos[1] - 15, f"B{pid}", color='red', fontsize=10))

wrong_line, = ax.plot([], [], 'y--', linewidth=2, label='Wrong Pass')
suggested_line, = ax.plot([], [], 'g-', linewidth=2, label='Suggested Pass')

def init():
    for pid, dot in dots.items():
        dot.set_data([], [])
    wrong_line.set_data([], [])
    suggested_line.set_data([], [])
    return list(dots.values()) + [wrong_line, suggested_line] + texts

def animate(i):
    for pid, dot in dots.items():
        if pid in team_a:
            dot.set_data(team_a[pid])
        else:
            dot.set_data(team_b[pid])

    if i >= 10:
        p1 = team_a[wrong_pass_from]
        p2 = team_a[wrong_pass_to]
        wrong_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    if i >= 20:
        p1 = team_a[wrong_pass_from]
        p2 = suggested_pos
        suggested_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    return list(dots.values()) + [wrong_line, suggested_line] + texts

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=40, interval=200, blit=True)

# Save as MP4
os.makedirs("tactical_animations", exist_ok=True)
ani.save("tactical_animations/tactical_suggestion.mp4", writer='ffmpeg', fps=2)

print("✅ Tactical suggestion video saved as 'tactical_animations/tactical_suggestion.mp4'")
