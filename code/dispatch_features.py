"""Rol C - Planning: gozlemden cikarilan mesafe ve durum ozellikleri."""
import numpy as np
from collections import deque

# grid hucre kodlari (gozlemde gorduk): 0=bos, 1=yasak, 3=sarj/hub
FREE, NOFLY, CHARGER = 0, 1, 3
def bfs_distances(grid, source):
    """source hucresinden her hucreye en kisa yol uzakligi (yasak hucreler dolasilarak).
    Ulasilamayan / yasak hucreler np.inf kalir."""
    H, W = grid.shape
    dist = np.full((H, W), np.inf)        # her yer baslangicta sonsuz
    sx, sy = int(source[0]), int(source[1])
    dist[sx, sy] = 0.0                     # baslangic hucresi 0
    queue = deque([(sx, sy)])              # islenecek hucreler kuyrugu
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # yukari, asagi, sol, sag

    while queue:
        x, y = queue.popleft()
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            # 1) grid sinirlari icinde mi?
            # 2) yasak degil mi?  (grid[nx, ny] != NOFLY)
            # 3) daha once ziyaret edilmemis mi?  (dist[nx, ny] == np.inf)
            if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] != NOFLY and dist[nx, ny] == np.inf:
                dist[nx, ny] = dist[x, y] + 1.0
                queue.append((nx, ny))

    return dist