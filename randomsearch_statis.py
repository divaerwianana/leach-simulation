import streamlit as st
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time

def run():

    st.title("Random Search Statis")

    # =========================
    # INPUT MANUAL
    # =========================
    num_nodes = st.number_input("Jumlah Node", 10, 500, 100)

    area_x = st.number_input("Area X", 100, 2000, 1000)
    area_y = st.number_input("Area Y", 100, 2000, 1000)

    initial_energy = st.number_input("Energi Awal", 0.1, 1.0, 0.5)

    bs_x = st.number_input("Base Station X", 0, int(area_x), int(area_x/2))
    bs_y = st.number_input("Base Station Y", 0, int(area_y), int(area_y/2))
    base_station = (bs_x, bs_y)

    frames = st.number_input("Jumlah Frame", 1000, 50000, 5000)
    round_interval = st.number_input("Round Interval", 10, 500, 100)
    speed = st.slider("Kecepatan Animasi", 0.001, 0.1, 0.01)

    # =========================
    # RUN
    # =========================
    if st.button("Run Simulation"):

        # PARAMETER
        comm_range = 80.0
        E_elec = 50e-9
        E_amp = 100e-12
        packet_size = 4000

        # =========================
        # INISIALISASI
        # =========================
        pos = np.random.rand(num_nodes, 2) * [area_x, area_y]
        energy = np.ones(num_nodes) * initial_energy

        FND, HND, LND = None, None, None
        total_packets_sent = 0
        total_packets_received = 0

        round_data = {
            "round": [],
            "alive_nodes": [],
            "total_energy": [],
            "avg_member_dist": [],
            "packets_sent": [],
            "packets_received": [],
            "PDR": [],
            "packet_loss": []
        }

        CH_index = []
        assigned_ch = None

        # =========================
        # FUNCTION
        # =========================
        def elect_cluster_heads(alive_nodes):
            p = 0.05
            num_ch = max(1, int(p * len(alive_nodes)))
            return random.sample(alive_nodes, num_ch)

        def assign_clusters(CH_index, positions):
            dists = np.linalg.norm(
                positions[:, None, :] - positions[CH_index][None, :, :],
                axis=2
            )
            nearest_idx = np.argmin(dists, axis=1)
            nearest_ch = np.array(CH_index)[nearest_idx]
            nearest_dist = dists[np.arange(len(positions)), nearest_idx]
            return nearest_ch, nearest_dist

        # =========================
        # TEMPAT VISUAL
        # =========================
        placeholder = st.empty()

        # =========================
        # SIMULASI PER FRAME
        # =========================
        for frame in range(frames):

            # =====================
            # LOGIKA PER ROUND
            # =====================
            if frame % round_interval == 0:

                round_id = frame // round_interval
                alive_nodes = np.where(energy > 0)[0].tolist()

                if len(alive_nodes) == 0:
                    if LND is None:
                        LND = round_id
                    break

                CH_index = elect_cluster_heads(alive_nodes)
                assigned_ch, dist_to_ch = assign_clusters(CH_index, pos)

                packets_sent = 0
                packets_received = 0

                for i in range(num_nodes):

                    if energy[i] <= 0:
                        continue

                    # CH → BS
                    if i in CH_index:
                        d_bs = np.linalg.norm(pos[i] - np.array(base_station))
                        energy_tx = (E_elec * packet_size) + (E_amp * packet_size * (d_bs ** 2))

                        energy[i] -= energy_tx
                        packets_sent += 1
                        packets_received += 1

                    # Member → CH
                    else:
                        ch = assigned_ch[i]
                        d = dist_to_ch[i]

                        if d <= comm_range and energy[ch] > 0:
                            energy_tx = (E_elec * packet_size) + (E_amp * packet_size * (d ** 2))
                            energy_rx = (E_elec * packet_size)

                            energy[i] -= energy_tx
                            energy[ch] -= energy_rx

                            packets_sent += 1
                            packets_received += 1
                        else:
                            packets_sent += 1

                energy[energy < 0] = 0

                total_packets_sent += packets_sent
                total_packets_received += packets_received

                alive_count = np.count_nonzero(energy > 0)
                total_energy = np.sum(energy)
                PDR = packets_received / packets_sent if packets_sent > 0 else 0

                round_data["round"].append(round_id)
                round_data["alive_nodes"].append(alive_count)
                round_data["total_energy"].append(total_energy)
                round_data["avg_member_dist"].append(
                    np.mean(dist_to_ch[energy > 0]) if alive_count > 0 else 0
                )
                round_data["packets_sent"].append(total_packets_sent)
                round_data["packets_received"].append(total_packets_received)
                round_data["PDR"].append(PDR)
                round_data["packet_loss"].append(1 - PDR)

                if FND is None and alive_count < num_nodes:
                    FND = round_id

                if HND is None and alive_count <= num_nodes / 2:
                    HND = round_id

            # =====================
            # VISUAL PER FRAME
            # =====================
            fig, ax = plt.subplots(figsize=(6,6))

            colors = [
                'black' if energy[i] <= 0
                else 'red' if i in CH_index
                else 'dodgerblue'
                for i in range(num_nodes)
            ]

            ax.scatter(pos[:,0], pos[:,1], c=colors)

            # Base Station
            ax.scatter(*base_station, marker='s', c='green')

            # Garis koneksi
            if assigned_ch is not None:
                for i in range(num_nodes):
                    if energy[i] > 0 and i not in CH_index:
                        ch = assigned_ch[i]
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
            time.sleep(speed)

        # =========================
        # OUTPUT (LINE CHART 🔥)
        # =========================
        st.subheader("Hasil")

        st.write("FND:", FND)
        st.write("HND:", HND)
        st.write("LND:", LND)

        df = pd.DataFrame(round_data)

        st.line_chart(df.set_index("round")["alive_nodes"])
        st.line_chart(df.set_index("round")["total_energy"])
        st.line_chart(df.set_index("round")["PDR"])
        st.line_chart(df.set_index("round")["packet_loss"])