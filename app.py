import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Необходимо для 3D графики
import numpy as np

from data import get_dataset, FEATURE_NAMES
from model import KMeansClustering, MaximinClustering

# Киберпанк тема
ctk.set_appearance_mode("Dark")
COLOR_BG = "#0D0D12"
COLOR_PANEL = "#16161F"
COLOR_ACCENT = "#00E676"  # Неоновый зеленый
COLOR_TEXT = "#E0E0E0"
CLUSTER_COLORS = ['#00E676', '#FF007F', '#00D4FF', '#FFD700', '#B026FF', '#FF5722']


class ClusterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Кластерный Анализ 3D | Сеть 5G (Вариант 12)")
        self.geometry("1100x700")
        self.configure(fg_color=COLOR_BG)

        self.X, self.names = get_dataset()
        self.chart_canvas = None

        self.setup_ui()

    def setup_ui(self):
        # Левая панель: Управление
        left_panel = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=15, width=320)
        left_panel.pack(side="left", fill="y", padx=15, pady=15)
        left_panel.pack_propagate(False)

        ctk.CTkLabel(left_panel, text="Параметры алгоритма", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLOR_TEXT).pack(pady=20)

        self.algo_var = ctk.StringVar(value="kmeans")

        rb_kmeans = ctk.CTkRadioButton(left_panel, text="Метод К-средних", variable=self.algo_var, value="kmeans",
                                       fg_color=COLOR_ACCENT, text_color=COLOR_TEXT, font=ctk.CTkFont(size=14))
        rb_kmeans.pack(anchor="w", padx=25, pady=10)

        rb_maximin = ctk.CTkRadioButton(left_panel, text="Метод Максимина", variable=self.algo_var, value="maximin",
                                        fg_color=COLOR_ACCENT, text_color=COLOR_TEXT, font=ctk.CTkFont(size=14))
        rb_maximin.pack(anchor="w", padx=25, pady=10)

        # Слайдер для K-средних
        self.lbl_k = ctk.CTkLabel(left_panel, text="Количество кластеров (K): 3", text_color="#A0A0A0")
        self.lbl_k.pack(pady=(20, 5))
        self.slider_k = ctk.CTkSlider(left_panel, from_=2, to=5, number_of_steps=3, button_color=COLOR_ACCENT,
                                      progress_color=COLOR_ACCENT)
        self.slider_k.set(3)
        self.slider_k.configure(command=lambda val: self.lbl_k.configure(text=f"Количество кластеров (K): {int(val)}"))
        self.slider_k.pack(fill="x", padx=25, pady=5)

        btn_run = ctk.CTkButton(left_panel, text="Кластеризовать", font=ctk.CTkFont(size=15, weight="bold"),
                                fg_color=COLOR_ACCENT, text_color="#000000", hover_color="#00C853", height=45,
                                command=self.run_clustering)
        btn_run.pack(fill="x", padx=25, pady=30)

        # Вывод результатов текстом
        self.res_frame = ctk.CTkFrame(left_panel, fg_color="#0D0D12", corner_radius=10)
        self.res_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.lbl_stats = ctk.CTkLabel(self.res_frame, text="Ожидание запуска...", justify="left", text_color=COLOR_TEXT,
                                      font=ctk.CTkFont(size=13))
        self.lbl_stats.pack(anchor="nw", padx=15, pady=15)

        # Правая панель: 3D График
        right_panel = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        ctk.CTkLabel(right_panel, text="3D Визуализация признакового пространства",
                     font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT).pack(pady=(15, 5))

        self.graph_container = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.graph_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.run_clustering()  # Запуск по умолчанию

    def run_clustering(self):
        algo = self.algo_var.get()

        if algo == "kmeans":
            k = int(self.slider_k.get())
            model = KMeansClustering(n_clusters=k)
            labels, centroids = model.fit(self.X)
            complexity = f"O(Итерации * K * N * M)"
        else:
            model = MaximinClustering()
            labels, centroids = model.fit(self.X)
            k = len(centroids)
            complexity = f"O(Итерации * K * N * M)"

        # Обновление статистики
        stats = (
            f"Метод: {'К-средних' if algo == 'kmeans' else 'Максимина'}\n\n"
            f"Объектов (N): {len(self.X)}\n"
            f"Признаков (M): {self.X.shape[1]}\n"
            f"Найдено кластеров (K): {k}\n"
            f"Итераций: {model.iterations}\n\n"
            f"Вычисл. сложность:\n{complexity}"
        )
        self.lbl_stats.configure(text=stats)

        self.plot_3d(labels, centroids)

    def plot_3d(self, labels, centroids):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(7, 6), dpi=100)
        fig.patch.set_facecolor(COLOR_PANEL)

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(COLOR_PANEL)

        # Отрисовка точек по кластерам
        unique_labels = np.unique(labels)
        for idx, cluster_id in enumerate(unique_labels):
            color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
            cluster_points = self.X[labels == cluster_id]

            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                       c=color, s=80, edgecolors='w', alpha=0.8, label=f'Кластер {cluster_id + 1}')

        # Отрисовка центроидов
        if len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='#FFFFFF', marker='X', s=200, edgecolors='#000000', label='Прототипы (Центры)')

        # Настройка осей (фон осей делаем прозрачным для красоты)
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_xlabel(FEATURE_NAMES[0], color='#A0A0A0')
        ax.set_ylabel(FEATURE_NAMES[1], color='#A0A0A0')
        ax.set_zlabel(FEATURE_NAMES[2], color='#A0A0A0')
        ax.tick_params(colors='#A0A0A0')

        ax.legend(loc='upper right', facecolor=COLOR_PANEL, edgecolor='#333333')
        fig.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.graph_container)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = ClusterApp()
    app.mainloop()