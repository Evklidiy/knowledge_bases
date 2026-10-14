import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from data import PROJECTS_DATA, OPTIMIZATION_GOALS, EXPERT_EVALUATIONS
import model

# Настройка современного темного дизайна
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

COLOR_BG = "#121212"
COLOR_SURFACE = "#1E1E1E"
COLOR_ACCENT = "#00BFA5"
COLOR_TEXT = "#E0E0E0"


class DecisionDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Анализатор ЭЛЕКТРА | Выбор проекта строительства")
        self.geometry("1050x700")
        self.configure(fg_color=COLOR_BG)

        # Подготовка данных
        self.raw_data = np.array([item["metrics"] for item in PROJECTS_DATA])
        self.project_names = np.array([item["id"] for item in PROJECTS_DATA])
        self.chart_canvas = None

        self.setup_ui()

    def setup_ui(self):
        # Верхняя панель (Контрол-бар)
        top_bar = ctk.CTkFrame(self, fg_color=COLOR_SURFACE, corner_radius=8, height=100)
        top_bar.pack(side="top", fill="x", padx=15, pady=15)

        # Веса критериев
        weights = model.compute_criteria_weights(EXPERT_EVALUATIONS)
        lbl_weights = ctk.CTkLabel(
            top_bar,
            text=f"Веса (Метод рангов):\nПрибыль: {weights[0]:.2f} | Раб. места: {weights[1]:.2f} | Развитие: {weights[2]:.2f}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_ACCENT,
            justify="left"
        )
        lbl_weights.pack(side="left", padx=20, pady=15)

        # Слайдеры управления порогами
        slider_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        slider_frame.pack(side="right", fill="x", expand=True, padx=20)

        self.val_c = ctk.StringVar(value="c* (Согласие): 0.50")
        ctk.CTkLabel(slider_frame, textvariable=self.val_c, text_color=COLOR_TEXT).grid(row=0, column=0, padx=10,
                                                                                        sticky="e")
        self.slider_c = ctk.CTkSlider(slider_frame, from_=0.0, to=1.0, command=self.on_slider_change,
                                      button_color=COLOR_ACCENT)
        self.slider_c.set(0.5)
        self.slider_c.grid(row=0, column=1, padx=10, pady=5)

        self.val_d = ctk.StringVar(value="d* (Несогласие): 0.50")
        ctk.CTkLabel(slider_frame, textvariable=self.val_d, text_color=COLOR_TEXT).grid(row=1, column=0, padx=10,
                                                                                        sticky="e")
        self.slider_d = ctk.CTkSlider(slider_frame, from_=0.0, to=1.0, command=self.on_slider_change,
                                      button_color="#F06292")
        self.slider_d.set(0.5)
        self.slider_d.grid(row=1, column=1, padx=10, pady=5)

        # Основная рабочая область (Сетка)
        main_grid = ctk.CTkFrame(self, fg_color="transparent")
        main_grid.pack(side="bottom", fill="both", expand=True, padx=15, pady=(0, 15))
        main_grid.columnconfigure(0, weight=1)
        main_grid.columnconfigure(1, weight=1)

        # Левая часть: Таблица
        table_container = ctk.CTkFrame(main_grid, fg_color=COLOR_SURFACE, corner_radius=8)
        table_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(table_container, text="Результаты анализа", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        self.table_content = ctk.CTkScrollableFrame(table_container, fg_color="transparent")
        self.table_content.pack(fill="both", expand=True, padx=5, pady=5)

        # Правая часть: График
        self.graph_container = ctk.CTkFrame(main_grid, fg_color=COLOR_SURFACE, corner_radius=8)
        self.graph_container.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.refresh_dashboard()

    def on_slider_change(self, _=None):
        self.val_c.set(f"c* (Согласие): {self.slider_c.get():.2f}")
        self.val_d.set(f"d* (Несогласие): {self.slider_d.get():.2f}")
        self.refresh_dashboard()

    def refresh_dashboard(self):
        c_th = self.slider_c.get()
        d_th = self.slider_d.get()

        # Расчет математики
        norm_mat = model.build_normalized_matrix(self.raw_data, OPTIMIZATION_GOALS)
        pareto_mask = model.get_pareto_front(norm_mat)
        weights = model.compute_criteria_weights(EXPERT_EVALUATIONS)
        c_vals, d_vals, core_mask = model.execute_electre(norm_mat, weights, c_th, d_th)

        self.update_table(pareto_mask, core_mask, c_vals, d_vals)
        self.update_chart(pareto_mask, c_vals, d_vals, c_th, d_th)

    def update_table(self, pareto_mask, core_mask, c_vals, d_vals):
        for w in self.table_content.winfo_children():
            w.destroy()

        headers = ["Проект", "Парето", "Инд. c", "Инд. d", "Ядро (Выбор)"]
        for col, h in enumerate(headers):
            ctk.CTkLabel(self.table_content, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=col, padx=12,
                                                                                           pady=10)

        for i, proj in enumerate(self.project_names):
            is_pareto = "✓" if pareto_mask[i] else "×"
            is_core = "ОПТИМУМ" if core_mask[i] and pareto_mask[i] else "-"

            c_str = f"{c_vals[i]:.3f}" if pareto_mask[i] else "—"
            d_str = f"{d_vals[i]:.3f}" if pareto_mask[i] else "—"

            color = COLOR_ACCENT if is_core == "ОПТИМУМ" else (COLOR_TEXT if pareto_mask[i] else "#555555")
            weight = "bold" if is_core == "ОПТИМУМ" else "normal"

            font = ctk.CTkFont(weight=weight)
            ctk.CTkLabel(self.table_content, text=proj, text_color=color, font=font).grid(row=i + 1, column=0, pady=6)
            ctk.CTkLabel(self.table_content, text=is_pareto, text_color=color, font=font).grid(row=i + 1, column=1,
                                                                                               pady=6)
            ctk.CTkLabel(self.table_content, text=c_str, text_color=color, font=font).grid(row=i + 1, column=2, pady=6)
            ctk.CTkLabel(self.table_content, text=d_str, text_color=color, font=font).grid(row=i + 1, column=3, pady=6)
            ctk.CTkLabel(self.table_content, text=is_core, text_color=color, font=font).grid(row=i + 1, column=4,
                                                                                             pady=6)

    def update_chart(self, pareto_mask, c_vals, d_vals, c_th, d_th):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.patch.set_facecolor(COLOR_SURFACE)
        ax.set_facecolor(COLOR_BG)

        # Выводим только альтернативы из множества Парето
        c_p = c_vals[pareto_mask]
        d_p = d_vals[pareto_mask]
        labels = self.project_names[pareto_mask]

        ax.scatter(c_p, d_p, color=COLOR_ACCENT, s=100, edgecolors='white', zorder=5)

        for i, label in enumerate(labels):
            ax.annotate(label, (c_p[i], d_p[i]), xytext=(7, 7), textcoords='offset points', color='white',
                        weight='bold')

        # Построение пороговых границ
        ax.axvline(x=c_th, color='#00BFA5', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axhline(y=d_th, color='#F06292', linestyle='--', linewidth=1.5, alpha=0.8)

        # Подсветка зоны ядра (правый нижний квадрант)
        ax.axvspan(c_th, 1.1, ymin=0, ymax=d_th / 1.1 if d_th <= 1.1 else 1, color='#00BFA5', alpha=0.1)

        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Индекс согласия (c) →", color='#9E9E9E', fontsize=10)
        ax.set_ylabel("Индекс несогласия (d) ↓", color='#9E9E9E', fontsize=10)
        ax.set_title("Карта Парето-оптимальных проектов", color='white', pad=15)

        ax.tick_params(colors='#9E9E9E')
        for spine in ax.spines.values():
            spine.set_color('#333333')

        ax.grid(True, color='#333333', linestyle='-', linewidth=0.5)
        fig.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.graph_container)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


if __name__ == "__main__":
    app = DecisionDashboard()
    app.mainloop()