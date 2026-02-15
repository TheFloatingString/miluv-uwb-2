import matplotlib.pyplot as plt
import pandas as pd

df_los = pd.read_csv(
    "data/processed_data/miluv-random_1-ifo001-uwb_cir_0.csv"
)  # 0,2,5: LOS
df_nlos_styrofoam = pd.read_csv(
    "data/processed_data/miluv-random_1-ifo001-uwb_cir_1.csv"
)  # 1: styrofoam
df_nlos_plastic = pd.read_csv(
    "data/processed_data/miluv-random_1-ifo001-uwb_cir_3.csv"
)  # 3: plastic
df_nlos_wood = pd.read_csv(
    "data/processed_data/miluv-random_1-ifo001-uwb_cir_4.csv"
)  # 4: wood

plt.suptitle("NLOS and LOS CIR Sample Responses from the MILUV Dataset")
plt.subplot(2, 2, 1)
plt.title("LOS Sample")
plt.plot(eval(df_los["cir"].values[0]), label="CIR")
plt.legend(loc="upper left")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.ylim(0, 10000)
plt.grid()
plt.subplot(2, 2, 2)
plt.title("Polyesterene NLOS Sample")
plt.plot(eval(df_nlos_styrofoam["cir"].values[0]), label="CIR")
plt.legend(loc="upper left")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.ylim(0, 10000)
plt.grid()
plt.subplot(2, 2, 3)
plt.title("Acrylic Plastic NLOS Sample")
plt.plot(eval(df_nlos_plastic["cir"].values[0]), label="CIR")
plt.legend(loc="upper left")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.ylim(0, 10000)
plt.grid()
plt.subplot(2, 2, 4)
plt.title("Wood NLOS Sample")
plt.plot(eval(df_nlos_wood["cir"].values[0]), label="CIR")
plt.legend(loc="upper left")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.ylim(0, 10000)
plt.grid()
plt.tight_layout()

plt.savefig("cir_plot.png")
