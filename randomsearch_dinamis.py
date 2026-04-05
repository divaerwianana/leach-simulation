def run():
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import pandas as pd
    import time

    st.title("Simulasi Random Search-LEACH Dinamis")

    # =============================
    # INPUT (MANUAL)
    # =============================
    num_nodes = int(st.number_input("Jumlah Node", value=100, min_value=1))

    initial_energy = st.number_input("Energi Awal", value=0.5, min_value=0.1)

    area_x = int(st.number_input("Luas Area X", value=1000, min_value=100))
    area_y = int(st.number_input("Luas Area Y", value=1000, min_value=100))

    bs_x = int(st.number_input("Base Station X", value=area_x//2, min_value=0, max_value=area_x))
    bs_y = int(st.number_input("Base Station Y", value=area_y//2, min_value=0, max_value=area_y))

    base_station = np.array([bs_x, bs_y])

    # =============================
    # PARAMETER
    # =============================
    comm_range = 80
    round_interval = 20
    frames = 10000

    E_elec = 50e-9
    E_amp = 100e-12
    packet_size = 4000

    num_bees = 20
    limit = 10
    max_iter = 20
    p = 0.1

    # =============================
    # INIT
    # =============================
    pos = np.random.rand(num_nodes,2) * [area_x,area_y]
    velocity = (np.random.rand(num_nodes,2)-0.5)*2
    energy = np.ones(num_nodes)*initial_energy

    # =============================
    # DATA SIMPAN
    # =============================
    round_data = {
        "round":[],
        "alive_nodes":[],
        "total_energy":[],
        "PDR":[]
    }

    FND = HND = LND = None

    # =============================
    # Random Search
    # =============================
    def elect_cluster_heads():
        alive = [i for i in range(num_nodes) if energy[i] > 0]
        if not alive:
            return []

        # jumlah CH target sama seperti sebelumnya
        n_ch = max(5, int(p * len(alive)))

        # pilih node secara acak
        if len(alive) <= n_ch:
            return alive
        else:
            return random.sample(alive, n_ch)
    
    # =============================
    # CLUSTER
    # =============================
    def assign_clusters(CH):
        if not CH:
            return np.zeros(num_nodes,dtype=int), np.full(num_nodes,np.inf)

        d = np.linalg.norm(pos[:,None,:]-pos[CH][None,:,:],axis=2)
        idx = np.argmin(d,axis=1)

        return np.array(CH)[idx], d[np.arange(num_nodes),idx]

    # =============================
    # RUN
    # =============================
    if st.button("Mulai Simulasi"):

        plot_area = st.empty()

        CH_index = []
        assigned_ch = np.zeros(num_nodes)

        for frame in range(frames):

            # ================= GERAK NODE =================
            for i in range(num_nodes):
                if energy[i] > 0:
                    pos[i] += velocity[i]*1.5

            pos[:] = np.clip(pos,0,[area_x,area_y])

            for i in range(num_nodes):
                if energy[i] <= 0: continue
                if pos[i,0]<=0 or pos[i,0]>=area_x:
                    velocity[i,0]*=-1
                if pos[i,1]<=0 or pos[i,1]>=area_y:
                    velocity[i,1]*=-1

            # ================= ROUND =================
            if frame % round_interval == 0:

                r = frame // round_interval
                alive_nodes = np.where(energy>0)[0]

                if len(alive_nodes)==0:
                    LND = r
                    break

                if FND is None and len(alive_nodes)<num_nodes:
                    FND = r
                if HND is None and len(alive_nodes)<=num_nodes/2:
                    HND = r

                CH_index = elect_cluster_heads()

                alive_nodes = list(np.where(energy > 0)[0])
                target_ch = max(5, int(p * len(alive_nodes)))

                # ================= TAMBAH CH JIKA KURANG =================
                if len(CH_index) < target_ch:
                    tambahan = list(set(alive_nodes) - set(CH_index))

                    if len(tambahan) > 0:
                        extra = random.sample(
                            tambahan,
                            min(target_ch - len(CH_index), len(tambahan))
                        )
                        CH_index = list(set(CH_index + extra))

                # ================= FIX: JANGAN SAMPAI KOSONG =================
                if len(CH_index) == 0 and len(alive_nodes) > 0:
                    CH_index = random.sample(alive_nodes, min(3, len(alive_nodes)))

                assigned_ch, dist = assign_clusters(CH_index)

                packets_sent=0
                packets_received=0

                for i in range(num_nodes):

                    if energy[i]<=0: continue

                    if i in CH_index:
                        d_bs=np.linalg.norm(pos[i]-base_station)
                        energy[i]-=(E_elec*packet_size)+(E_amp*packet_size*(d_bs**2))
                        packets_sent+=1
                        packets_received+=1
                    else:
                        ch = assigned_ch[i]
                        if dist[i]<=comm_range and energy[ch]>0:
                            energy[i]-=(E_elec*packet_size)+(E_amp*packet_size*(dist[i]**2))
                            energy[ch]-=(E_elec*packet_size)
                            packets_received+=1
                        packets_sent+=1

                energy[energy<0]=0

                total_energy=np.sum(energy)
                PDR = packets_received/packets_sent if packets_sent>0 else 0

                round_data["round"].append(r)
                round_data["alive_nodes"].append(len(alive_nodes))
                round_data["total_energy"].append(total_energy)
                round_data["PDR"].append(PDR)

            # ================= PLOT ANIMASI =================
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_xlim(0,area_x)
            ax.set_ylim(0,area_y)

            # ================= GARIS NODE → CH =================
            for i in range(num_nodes):
                if energy[i] <= 0 or i in CH_index:
                    continue

                ch = assigned_ch[i]

                if energy[ch] > 0:
                    ax.plot(
                        [pos[i,0], pos[ch,0]],
                        [pos[i,1], pos[ch,1]],
                        color='gray',
                        linewidth=0.5,
                        alpha=0.3
                    )

            # ================= GARIS CH → BS =================
            for ch in CH_index:
                if energy[ch] <= 0:
                    continue

                ax.plot(
                    [pos[ch,0], base_station[0]],
                    [pos[ch,1], base_station[1]],
                    color='orange',
                    linewidth=1.2
                )

            # ================= NODE =================
            colors = [
                'black' if energy[i]<=0
                else 'red' if i in CH_index
                else 'blue'
                for i in range(num_nodes)
            ]

            ax.scatter(pos[:,0],pos[:,1],c=colors,s=30)

            # BASE STATION
            ax.scatter(base_station[0],base_station[1],
                    c='green',s=100,marker='s')

            alive_count = np.count_nonzero(energy > 0)
            total_energy = np.sum(energy)

            ax.set_title(
                f"Frame {frame} | Alive: {alive_count}/{num_nodes} | Energy: {total_energy:.2f}"
            )
            plot_area.pyplot(fig)
            plt.close(fig)

            time.sleep(0.01)

        # =============================
        # HASIL
        # =============================
        st.subheader("Hasil Simulasi")
        st.write(f"FND: {FND}")
        st.write(f"HND: {HND}")
        st.write(f"LND: {LND}")

        df = pd.DataFrame(round_data)

        # Grafik
        st.subheader("Grafik Alive Nodes")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["round"], df["alive_nodes"])
        st.pyplot(fig1)

        st.subheader("Grafik Total Energy")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["round"], df["total_energy"])
        st.pyplot(fig2)

        st.subheader("Grafik PDR")
        fig3, ax3 = plt.subplots()
        ax3.plot(df["round"], df["PDR"])
        st.pyplot(fig3)

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "hasil_rs_dinamis.csv")
