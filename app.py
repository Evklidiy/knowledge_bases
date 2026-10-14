import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from data import get_training_dataset, FEATURE_NAMES
from model import MinDistanceClassifier, PerceptronClassifier

# Тема оформления (Deep Purple)
ctk.set_appearance_mode("Dark")
COLOR_BG = "#131018"
COLOR_PANEL = "#1E1926"
COLOR_ACCENT = "#9D4EDD"
COLOR_CLASS_1 = "#4CC9F0"  # Голубой (Норма)
COLOR_CLASS_2 = "#F72585"  # Розовый (Аномалия)
COLOR_TEXT = "#F8F9FA"


class RecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание образов | Анализатор сети (Вариант 12)")
        self.geometry("1100x700")
        self.configure(fg_color=COLOR_BG)

        # Инициализация и обучение моделей
        self.X_train, self.y_train = get_training_dataset()

        self.min_dist_model = MinDistanceClassifier()
        self.min_dist_model.fit(self.X_train, self.y_train)

        self.perceptron_model = PerceptronClassifier()
        self.perceptron_model.fit(self.X_train, self.y_train)

        self.sliders = []
        self.chart_canvas = None
        self.setup_ui()

    def setup_ui(self):
        # Левая панель: Ввод данных распознаваемого объекта
        left_panel = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=12, width=320)
        left_panel.pack(side="left", fill="y", padx=15, pady=15)
        left_panel.pack_propagate(False)

        ctk.CTkLabel(left_panel, text="Признаки объекта", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLOR_TEXT).pack(pady=(20, 20))

        for name in FEATURE_NAMES:
            frame = ctk.CTkFrame(left_panel, fg_color="transparent")
            frame.pack(fill="x", padx=20, pady=8)

            lbl = ctk.CTkLabel(frame, text=f"{name}: 0.50", text_color="#A0A0A0", font=ctk.CTkFont(size=13))
            lbl.pack(anchor="w")

            slider = ctk.CTkSlider(frame, from_=0.0, to=1.0, button_color=COLOR_ACCENT, progress_color=COLOR_ACCENT)
            slider.set(0.5)
            slider.configure(command=lambda val, l=lbl, n=name: l.configure(text=f"{n}: {val:.2f}"))
            slider.pack(fill="x", pady=(2, 0))
            self.sliders.append(slider)

        self.btn_recognize = ctk.CTkButton(
            left_panel, text="Распознать класс", font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLOR_ACCENT, hover_color="#7B2CBF", height=45, command=self.perform_recognition
        )
        self.btn_recognize.pack(fill="x", padx=20, pady=(30, 10))

        # Фрейм для вывода текстовых результатов
        self.res_frame = ctk.CTkFrame(left_panel, fg_color="#131018", corner_radius=8)
        self.res_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.lbl_res_md = ctk.CTkLabel(self.res_frame, text="", justify="left", text_color=COLOR_TEXT,
                                       font=ctk.CTkFont(size=13))
        self.lbl_res_md.pack(anchor="w", padx=10, pady=(10, 5))

        self.lbl_res_pc = ctk.CTkLabel(self.res_frame, text="", justify="left", text_color=COLOR_TEXT,
                                       font=ctk.CTkFont(size=13))
        self.lbl_res_pc.pack(anchor="w", padx=10, pady=(5, 10))

        # Правая панель: Геометрическая интерпретация (График)
        right_panel = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=12)
        right_panel.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        ctk.CTkLabel(
            right_panel, text="Геометрическая интерпретация (Проекция на признаки X1 и X2)",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT
        ).pack(pady=(15, 5))

        self.graph_container = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.graph_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.perform_recognition()

    def perform_recognition(self):
        # Сбор вектора признаков с ползунков
        x_test = np.array([s.get() for s in self.sliders])

        # 1. Метод минимального расстояния
        c_md, details_md = self.min_dist_model.predict(x_test)
        md_text = f"МИН. РАССТОЯНИЕ:\nКласс: {c_md}\nD1: {details_md[1]:.3f}\nD2: {details_md[2]:.3f}"

        # 2. Метод восприятия
        c_pc, d12 = self.perceptron_model.predict(x_test)
        pc_text = f"ПЕРЦЕПТРОН:\nКласс: {c_pc}\nD12: {d12:.3f}\nИтераций обучения: {self.perceptron_model.iterations}"

        self.lbl_res_md.configure(text=md_text)
        self.lbl_res_pc.configure(text=pc_text)

        self.update_geometry_plot(x_test, c_md, c_pc)

    def update_geometry_plot(self, x_test, c_md, c_pc):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor(COLOR_PANEL)
        ax.set_facecolor(COLOR_BG)

        # Отрисовка обучающего множества (только X1 и X2)
        X1_c1 = self.X_train[self.y_train == 1][:, 0]
        X2_c1 = self.X_train[self.y_train == 1][:, 1]
        X1_c2 = self.X_train[self.y_train == 2][:, 0]
        X2_c2 = self.X_train[self.y_train == 2][:, 1]

        ax.scatter(X1_c1, X2_c1, c=COLOR_CLASS_1, label='Класс 1 (Норма)', s=70, edgecolors='white', zorder=3)
        ax.scatter(X1_c2, X2_c2, c=COLOR_CLASS_2, label='Класс 2 (Аномалия)', s=70, edgecolors='white', zorder=3)

        # Отрисовка тестового объекта
        test_color = COLOR_CLASS_1 if c_pc == 1 else COLOR_CLASS_2
        ax.scatter(x_test[0], x_test[1], c=test_color, marker='*', s=300, label='Распознаваемый объект',
                   edgecolors='yellow', zorder=4)

        # --- Проекция разделяющих границ на 2D ---
        x_vals = np.linspace(-0.1, 1.1, 100)

        # Граница Перцептрона: W1*X1 + W2*X2 + (W3*X3 + W4*X4 + W5*X5 + W0) = 0
        w = self.perceptron_model.w
        w0 = self.perceptron_model.w0
        C_pc = w[2] * x_test[2] + w[3] * x_test[3] + w[4] * x_test[4] + w0
        if w[1] != 0:
            y_vals_pc = -(w[0] / w[1]) * x_vals - (C_pc / w[1])
            ax.plot(x_vals, y_vals_pc, color='#E0AFFF', linestyle='--', linewidth=2, label='Граница Перцептрона')

        # Граница Мин. Расстояния: D1(X) = D2(X)
        p1 = self.min_dist_model.prototypes[1]
        p2 = self.min_dist_model.prototypes[2]
        w_md = 2 * (p1 - p2)
        w0_md = np.dot(p2, p2) - np.dot(p1, p1)
        C_md = w_md[2] * x_test[2] + w_md[3] * x_test[3] + w_md[4] * x_test[4] + w0_md
        if w_md[1] != 0:
            y_vals_md = -(w_md[0] / w_md[1]) * x_vals - (C_md / w_md[1])
            ax.plot(x_vals, y_vals_md, color='#5E60CE', linestyle=':', linewidth=2, label='Граница Мин. Расстояния')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(f"X1 ({FEATURE_NAMES[0]})", color='#A0A0A0')
        ax.set_ylabel(f"X2 ({FEATURE_NAMES[1]})", color='#A0A0A0')
        ax.tick_params(colors='#A0A0A0')

        for spine in ax.spines.values():
            spine.set_color('#333333')

        ax.grid(True, color='#2A2A35', linestyle='-', linewidth=0.5, zorder=0)
        ax.legend(loc='upper left', fontsize=9, facecolor=COLOR_PANEL, edgecolor='#333333')

        fig.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.graph_container)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = RecognitionApp()
    app.mainloop()