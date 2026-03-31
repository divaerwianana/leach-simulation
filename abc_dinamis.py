import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def run():

    st.title("LEACH-ABC Dynamic")

    # =====================================================
    # INPUT BEBAS
    # =====================================================
    num_nodes = st.number_input("Jumlah Node", 10, 1000, 100)

    area_x = st.number_input("Luas Area X", 100, 2000, 1000)
    area_y = st.number_input("Luas Area Y", 100, 2000, 1000)

    initial_energy = st.number_input("Energi Awal", 0.01, 1.0, 0.5)

    bs_x = st.number_input("Posisi BS X", 0, 2000, int(area_x/2))
    bs_y = st.number_input("Posisi BS Y", 0, 2000, int(area_y/2))
    base_station = (bs_x, bs_y)

    frames = st.number_input("Frames", 1000, 20000, 5000)
    round_interval = st.number_input("Round Interval", 10, 500, 100)

    speed = st.slider("Kecepatan Gerak Node", 0.01, 1.0, 0.1)

    # =====================================================
    # RUN
    # =====================================================
    if st.button("Run Simulation"):

        p = 0.1
        comm_range = 100
        energy_tx = 0.02
        energy_rx = 0.005

        num_bees = 20
        limit = 10
        max_iter = 20

        # INISIALISASI
        pos = np.random.rand(num_nodes, 2) * [area_x, area_y]
        velocity = (np.random.rand(num_nodes, 2) - 0.5) * 2
        energy = np.ones(num_nodes) * initial_energy

        FND, HND, LND = None, None, None

        round_data = {
            "round": [],
            "alive_nodes": [],
            "dead_nodes": [],
            "PDR": [],
            "avg_member_dist": []
        }

        CH_index, prev_CH = [], []
        nearest_ch = np.zeros(num_nodes, dtype=int)
        nearest_dist = np.zeros(num_nodes)

        # =====================================================
        # FUNCTION
        # =====================================================
        def fitness_ch(i):
            if energy[i] <= 0:
                return 0
            d_bs = np.linalg.norm(pos[i] - np.array(base_station))
            return energy[i] / (d_bs + 1e-6)

        def elect_cluster_heads(prev_CH):
            alive = [i for i in range(num_nodes) if energy[i] > 0]
            if not alive:
                return []

            food = random.sample(alive, min(num_bees, len(alive)))
            fit = [fitness_ch(i) for i in food]
            trial = [0] * len(food)

            for _ in range(max_iter):

                for i in range(len(food)):
                    cand = random.choice(alive)
                    if fitness_ch(cand) > fit[i]:
                        food[i], fit[i], trial[i] = cand, fitness_ch(cand), 0
                    else:
                        trial[i] += 1

                prob = np.array(fit)
                prob = prob / np.sum(prob) if np.sum(prob) != 0 else np.ones(len(prob)) / len(prob)

                for _ in range(len(food)):
                    i = np.random.choice(len(food), p=prob)
                    cand = random.choice(alive)
                    if fitness_ch(cand) > fit[i]:
                        food[i], fit[i], trial[i] = cand, fitness_ch(cand), 0
                    else:
                        trial[i] += 1

                for i in range(len(food)):
                    if trial[i] > limit:
                        food[i] = random.choice(alive)
                        fit[i] = fitness_ch(food[i])
                        trial[i] = 0

            n_ch = max(1, int(p * len(alive)))
            idx = np.argsort(fit)[-n_ch:]
            return list(set(food[i] for i in idx))

        def assign_clusters(CH):
            if not CH:
                return np.zeros(num_nodes, dtype=int), np.full(num_nodes, np.inf)

            d = np.linalg.norm(pos[:, None, :] - pos[CH][None, :, :], axis=2)
            idx = np.argmin(d, axis=1)

            return np.array(CH)[idx], d[np.arange(num_nodes), idx]

        # =====================================================
        # VISUAL
        # =====================================================
        placeholder = st.empty()

        # =====================================================
        # SIMULASI
        # =====================================================
        for frame in range(frames):

            # GERAK NODE (pakai slider speed 🔥)
            pos += velocity * speed
            pos[:] = np.clip(pos, 0, [area_x, area_y])

            for i in range(num_nodes):
                if pos[i,0]<=0 or pos[i,0]>=area_x:
                    velocity[i,0]*=-1
                if pos[i,1]<=0 or pos[i,1]>=area_y:
                    velocity[i,1]*=-1

            # ROUND
            if frame % round_interval == 0:

                round_id = frame // round_interval

                CH_index = elect_cluster_heads(prev_CH)
                prev_CH = CH_index.copy()

                nearest_ch, nearest_dist = assign_clusters(CH_index)

                for i in range(num_nodes):

                    if energy[i] <= 0:
                        continue

                    if i in CH_index:
                        d = np.linalg.norm(pos[i] - np.array(base_station))
                        d = min(d, 200)
                        energy[i] -= energy_tx * (1 + d / area_x)

                    else:
                        ch = nearest_ch[i]

                        if energy[ch] > 0 and nearest_dist[i] <= comm_range:
                            energy[i] -= energy_tx
                            energy[ch] -= energy_rx

                energy = np.maximum(energy, 0)

                alive = np.count_nonzero(energy > 0)
                dead = num_nodes - alive

                if FND is None and dead >= 1: FND = round_id
                if HND is None and dead >= num_nodes/2: HND = round_id
                if LND is None and dead == num_nodes: LND = round_id

                round_data["round"].append(round_id)
                round_data["alive_nodes"].append(alive)
                round_data["dead_nodes"].append(dead)
                round_data["PDR"].append(alive / num_nodes)
                round_data["avg_member_dist"].append(
                    np.mean(nearest_dist[energy > 0]) if alive > 0 else 0
                )

            # VISUAL UPDATE
            if frame % 20 == 0:

                fig, ax = plt.subplots(figsize=(6,6))

                colors = [
                    'black' if energy[i]<=0 else
                    'red' if i in CH_index else 'dodgerblue'
                    for i in range(num_nodes)
                ]

                ax.scatter(pos[:,0], pos[:,1], c=colors)

                ax.scatter(*base_station, marker='s', c='green')

                for i in range(num_nodes):
                    if energy[i] > 0 and i not in CH_index:
                        ch = nearest_ch[i]
                        if energy[ch] > 0:
                            ax.plot(
                                [pos[i,0], pos[ch,0]],
                                [pos[i,1], pos[ch,1]],
                                color='gray', alpha=0.3, linewidth=0.5
                            )

                for ch in CH_index:
                    if energy[ch] > 0:
                        ax.plot(
                            [pos[ch,0], base_station[0]],
                            [pos[ch,1], base_station[1]],
                            color='orange', linewidth=1
                        )

                ax.set_xlim(0, area_x)
                ax.set_ylim(0, area_y)
                ax.set_title(f"Frame {frame}")

                placeholder.pyplot(fig)

        # OUTPUT
        st.subheader("Hasil")
        st.write("FND:", FND)
        st.write("HND:", HND)
        st.write("LND:", LND)

        df = pd.DataFrame(round_data)

        st.line_chart(df[["alive_nodes","dead_nodes"]])
        st.line_chart(df["PDR"])
        st.line_chart(df["avg_member_dist"])