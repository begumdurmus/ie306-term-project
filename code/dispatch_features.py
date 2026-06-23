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

def nearest_hub_distance(grid):
    """Her hucreden en yakin hub/sarj hucresine BFS mesafesi."""
    hub_cells = list(zip(*np.where(grid == CHARGER)))
    H, W = grid.shape
    best = np.full((H, W), np.inf)        # her hucre icin en yakin hub mesafesi
    for hub in hub_cells:
        d = bfs_distances(grid, hub)       # bu hub'dan tum hucrelere mesafe
        best = np.minimum(best, d)         # hucre bazinda kucugunu tut
    return best

def bucket(x, edges):
    """x sayisinin kac siniri (edge) gectigini dondurur = kova numarasi.
    Ornek: bucket(0.5, [0.15, 0.30, 0.45, 0.60, 0.80]) -> 3"""
    b = 0
    for e in edges:
        if x >= e:
            b += 1
        else:
            break
    return b

# makro-aksiyonlar
ASSIGN, CHARGE, HOLD = 0, 1, 2

# kova sinirlari (her ozellik icin)
SOC_EDGES    = [0.15, 0.30, 0.45, 0.60, 0.80]   # 6 kova: soc
HUB_EDGES    = [2, 5, 9, 15]                      # 5 kova: en yakin hub mesafesi
MARGIN_EDGES = [0.0, 0.10, 0.30]                 # 4 kova: bitirme marji
URG_EDGES    = [5, 15, 30]                        # 4 kova: son tarihe kalan
PRESS_EDGES  = [2, 6, 12]                         # 4 kova: bekleyen siparis sayisi

def decision_context(obs, cfg, hub_field):
    """Gozlemi alir; odak drone'u secer ve durum kovalarini uretir.
    hub_field: nearest_hub_distance(grid) ciktisi (her hucre icin hub mesafesi).
    Doner: (state, macro_mask, assign_action, charge_action, noop_action, info)
    """
    drones = obs["drones"]
    orders = obs["orders"]
    grid   = obs["grid"]
    mask   = obs["action_mask"]
    NK = cfg.n_drones * cfg.k_max

    # 1) Maskeden bos drone'lari ve atanabilir siparis slotlarini bul
    assign_bits = mask[:NK].reshape(cfg.n_drones, cfg.k_max).astype(bool)
    idle_drones = np.where(assign_bits.any(axis=1))[0]
    order_slots = np.where(assign_bits.any(axis=0))[0]

    # 2) Atanacak (drone, siparis) cifti yoksa: sadece HOLD mumkun
    if len(idle_drones) == 0 or len(order_slots) == 0:
        macro_mask = np.array([False, False, True])   # sadece HOLD
        return None, macro_mask, -1, -1, cfg.noop_index, {}

    # 3) En yakin (drone, siparis) cifti = odak drone (greedy gibi)
    best_d, best_pair = np.inf, None
    for d in idle_drones:
        dpos = (drones[d, 0], drones[d, 1])
        dist_from_d = bfs_distances(grid, dpos)        # bu drone'dan tum hucrelere
        for s in order_slots:
            opos = (int(orders[s, 0]), int(orders[s, 1]))
            dd = dist_from_d[opos]                      # drone -> siparis baslangici
            if dd < best_d:
                best_d, best_pair = dd, (int(d), int(s))
    focal, slot = best_pair

    # buraya kadar: focal = odak drone, slot = ona en yakin siparis
    # (devami bir sonraki parcada: ozellikler + kovalar + makro maske)
   # 4) Odak drone'un ozellikleri
    soc      = float(drones[focal, 2])                    # batarya (0-1)
    fpos     = (int(drones[focal, 0]), int(drones[focal, 1]))
    opos     = (int(orders[slot, 0]), int(orders[slot, 1]))    # siparis baslangici
    dpos     = (int(orders[slot, 2]), int(orders[slot, 3]))    # siparis varisi
    age      = float(orders[slot, 4])                     # siparisin yasi

    d_pick   = best_d                                     # drone -> alis (zaten var)
    d_leg    = bfs_distances(grid, opos)[dpos]            # alis -> teslim
    d_drop_hub = hub_field[dpos]                          # teslim -> en yakin hub
    d_cur_hub  = hub_field[fpos]                          # drone -> en yakin hub

    # bu isi alip bitirip hub'a donmek icin gereken enerji
    job_energy = cfg.e_move * (d_pick + d_leg + d_drop_hub)
    margin   = soc - job_energy                           # marj (negatifse tehlike)
    urgency  = cfg.sla_steps - age                        # son tarihe kalan adim
    pressure = len(order_slots)                           # bekleyen siparis sayisi

    # 5) Surekli ozellikleri kovalara cevir -> ayrik durum
    state = (
        bucket(soc,       SOC_EDGES),
        bucket(d_cur_hub, HUB_EDGES),
        bucket(margin,    MARGIN_EDGES),
        bucket(urgency,   URG_EDGES),
        bucket(pressure,  PRESS_EDGES),
    )

    # 6) Hangi makro-aksiyonlar gecerli?
    can_charge = bool(mask[cfg.charge_index(focal)])      # odak drone sarja gidebilir mi
    macro_mask = np.array([True, can_charge, True])       # ASSIGN, CHARGE, HOLD

    # 7) Makro-aksiyonu env aksiyon numarasina cevirmek icin gereken bilgiler
    assign_action = cfg.assign_index(focal, slot)
    charge_action = cfg.charge_index(focal) if can_charge else -1
    info = dict(soc=soc, margin=margin, urgency=urgency,
                pressure=pressure, dist_hub=d_cur_hub, focal=focal)

    return state, macro_mask, assign_action, charge_action, cfg.noop_index, info