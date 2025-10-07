# ============================================
# NP Universe 3D — Bounded Search (TSP)
# "Não dá para simular todas as opções": n! é astronômico.
# Amostragem estocástica + 2-opt limitado (orçamento de busca).
# Exporta GIF (fallback MP4).
# ============================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from itertools import combinations

# -------------------- PARÂMETROS DIDÁTICOS --------------------
np.random.seed(7)
N_CITIES   = 18          # mais cidades => n! explode
RADIUS     = 1.0
DEPTH_Z    = 0.35
N_STARS    = 1200
N_RINGS    = 9

# Orçamento de busca (aqui está a "impossibilidade" de explorar tudo)
SAMPLES            = 220        # quantas rotas aleatórias vamos tentar
TWO_OPT_STEPS_MAX  = 80         # limite de passos 2-opt por amostra
FRAMES_PER_SAMPLE  = 6          # suavidade da animação por amostra

FPS   = 24
W, H  = 1080, 1080
TITLE = "NP Universe 3D — Bounded Search (TSP)"

# Cores
COLOR_CUR   = "#3ac1ff"   # rota da amostra atual (ciano)
COLOR_BEST  = "#00e676"   # melhor até agora (verde)
COLOR_CITY  = "#ff9f1a"
COLOR_RING  = (0.8, 0.8, 0.8, 0.25)

# -------------------- GEOMETRIA / DADOS --------------------
def random_cities(n, radius=1.0, depth=0.35):
    theta = 2*np.pi*np.random.rand(n)
    r = radius * np.sqrt(np.random.rand(n))
    x = r*np.cos(theta); y = r*np.sin(theta)
    z = (np.random.rand(n)-0.5) * 2 * depth * 0.5
    return np.c_[x, y, z]

cities = random_cities(N_CITIES, RADIUS, DEPTH_Z)

def length(tour, pts):
    p = pts[tour]
    return np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1))

def two_opt_once(tour, pts):
    n = len(tour)
    best = tour[:]; best_gain = 0.0
    for i, k in combinations(range(n), 2):
        if (k - i) % n in (0, 1, n-1):
            continue
        a, b = tour[i], tour[(i+1) % n]
        c, d = tour[k], tour[(k+1) % n]
        old = np.linalg.norm(pts[a]-pts[b]) + np.linalg.norm(pts[c]-pts[d])
        new = np.linalg.norm(pts[a]-pts[c]) + np.linalg.norm(pts[b]-pts[d])
        gain = old - new
        if gain > best_gain:
            best_gain = gain
            best = tour[:i+1] + list(reversed(tour[i+1:k+1])) + tour[k+1:]
    return best, best_gain

def limited_two_opt(tour, pts, steps_max):
    cur = tour[:]
    for _ in range(steps_max):
        cur, gain = two_opt_once(cur, pts)
        if gain <= 1e-12:
            break
    return cur

# -------------------- MEDIDAS DE COMPLEXIDADE --------------------
# log10(n!) via função gama: log10(n!) = log10(Gamma(n+1)) = lgamma(n+1)/ln(10)
def log10_factorial(n):
    return math.lgamma(n+1) / math.log(10)

LOG10_SPACE = log10_factorial(N_CITIES)          # log10(n!)
SPACE_SCI = 10**(LOG10_SPACE - int(LOG10_SPACE)) # mantissa
SPACE_STR = f"{SPACE_SCI:.3f}e{int(LOG10_SPACE)}"  # n! ~ a.eb

# -------------------- FIGURA 3D --------------------
plt.close("all")
fig = plt.figure(figsize=(W/100, H/100), dpi=100)
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor("black"); ax.set_facecolor("black")
ax.set_title(TITLE, fontsize=16, pad=18, color="white")

mx = RADIUS * 1.25
ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-DEPTH_Z, DEPTH_Z)
ax.set_axis_off()

# starfield
stars = (np.random.rand(N_STARS, 3) - 0.5)
stars[:,0] *= 2*mx; stars[:,1] *= 2*mx; stars[:,2] *= 2*DEPTH_Z*1.2
ax.scatter(stars[:,0], stars[:,1], stars[:,2], s=1, alpha=0.25, zorder=0)

# rings
ring_objs = []
phi = np.linspace(0, 2*np.pi, 300)
for i in range(1, N_RINGS+1):
    rr = RADIUS * i / N_RINGS
    rx, ry = rr*np.cos(phi), rr*np.sin(phi)
    rz = np.zeros_like(rx)
    (ring_line,) = ax.plot(rx, ry, rz, color=COLOR_RING, linewidth=1.0, zorder=1)
    ring_objs.append(ring_line)

# cities
ax.scatter(cities[:,0], cities[:,1], cities[:,2],
           s=40, c=COLOR_CITY, edgecolors='white', linewidths=0.7, zorder=5)
for i,(xi,yi,zi) in enumerate(cities):
    ax.text(xi, yi, zi, f"{i}", color="white", fontsize=7, ha='center', va='center', zorder=6)

# lines
line_current, = ax.plot([], [], [], color=COLOR_CUR, linewidth=2.2, zorder=4, alpha=0.95)
line_best,    = ax.plot([], [], [], color=COLOR_BEST, linewidth=2.6, zorder=4, alpha=0.0)

# HUD
txt1 = fig.text(0.02, 0.96, "", fontsize=12, color="white")
txt2 = fig.text(0.02, 0.92, "", fontsize=12, color="white")
txt3 = fig.text(0.02, 0.88, "", fontsize=12, color="white")
txt4 = fig.text(0.02, 0.06, "", fontsize=10, color="white")

# -------------------- ANIMAÇÃO (amostragem orçada) --------------------
TOTAL_FRAMES = SAMPLES * FRAMES_PER_SAMPLE
best_tour = None
best_cost = float("inf")

# pré-gerar tours aleatórios (sem enumerar: pura amostragem)
random_tours = [np.random.permutation(N_CITIES).tolist() for _ in range(SAMPLES)]

def ease_out_cubic(t): return 1 - (1 - t)**3

def partial_path(points, fraction):
    P = np.vstack([points, points[0]])
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    cum = np.cumsum(seg); total = cum[-1]
    L = total * max(0,min(1,fraction))
    k = np.searchsorted(cum, L)
    xs, ys, zs = list(P[:k+1,0]), list(P[:k+1,1]), list(P[:k+1,2])
    if k < len(seg):
        remain = L - (cum[k-1] if k>0 else 0)
        if seg[k]>0:
            t = remain/seg[k]
            xs.append(P[k,0] + t*(P[k+1,0]-P[k,0]))
            ys.append(P[k,1] + t*(P[k+1,1]-P[k,1]))
            zs.append(P[k,2] + t*(P[k+1,2]-P[k,2]))
    return xs, ys, zs

def update(frame):
    global best_tour, best_cost

    # rotação suave da câmera
    az = (frame * 360 / TOTAL_FRAMES) % 360
    el = 20 + 10 * math.sin(2*math.pi * frame / TOTAL_FRAMES)
    ax.view_init(elev=el, azim=az)

    # scan rings
    k = (frame // max(1, FPS//2)) % N_RINGS
    for i, ring in enumerate(ring_objs):
        ring.set_linewidth(1.0 + (1.2 if i == k else 0.0))
        ring.set_alpha(0.18 + (0.18 if i == k else 0.0))

    sample_idx = min(frame // FRAMES_PER_SAMPLE, SAMPLES - 1)
    within = (frame % FRAMES_PER_SAMPLE) / (FRAMES_PER_SAMPLE - 1 + 1e-9)
    frac = ease_out_cubic(within)

    # rota atual (random + 2-opt limitado)
    base_tour = random_tours[sample_idx]
    improved  = limited_two_opt(base_tour, cities, TWO_OPT_STEPS_MAX)

    # atualiza "melhor até agora"
    cost_now = length(improved, cities)
    if cost_now < best_cost:
        best_cost = cost_now
        best_tour = improved[:]

    # desenha amostra atual parcialmente (ciano)
    Pcur = cities[improved]
    xs, ys, zs = partial_path(Pcur, fraction=frac)
    line_current.set_data(xs, ys); line_current.set_3d_properties(zs)
    line_current.set_alpha(0.95)

    # desenha melhor rota completa (verde)
    if best_tour is not None:
        Pbest = np.vstack([cities[best_tour], cities[best_tour[0]]])
        line_best.set_data(Pbest[:,0], Pbest[:,1]); line_best.set_3d_properties(Pbest[:,2])
        line_best.set_alpha(0.95 if sample_idx>0 else 0.0)

    # métricas de "impossibilidade"
    explored = sample_idx + frac
    log10_explored = math.log10(max(explored, 1e-12))
    percent = 10**(log10_explored - LOG10_SPACE) * 100.0
    percent_str = f"{percent:.3e}%" if percent < 0.001 else f"{percent:.6f}%"

    txt1.set_text(f"Espaço de busca ~ n! ≈ {SPACE_STR} (n={N_CITIES})")
    txt2.set_text(f"Amostras realizadas: {int(explored):>4d} / {SAMPLES}  —  Exploradas: ~{percent_str}")
    txt3.set_text(f"Melhor custo até agora: {best_cost:.1f}")
    txt4.set_text("Mensagem: não exploramos tudo — apenas amostramos e melhoramos localmente (2-opt limitado). "
                  "É assim que lidamos com NP: aproximação sob orçamento.")

    return (line_current, line_best, *ring_objs)

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=False)

# -------------------- SALVAR --------------------
try:
    anim.save("np_universe_budget.gif", writer=PillowWriter(fps=FPS))
    print("GIF salvo: np_universe_budget.gif")
except Exception as e:
    print("Falha no GIF, tentando MP4...", e)
    try:
        anim.save("np_universe_budget.mp4", fps=FPS, extra_args=['-vcodec','libx264'])
        print("MP4 salvo: np_universe_budget.mp4")
    except Exception as e2:
        print("Falha no MP4 também:", e2)
