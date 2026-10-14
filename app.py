import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import numpy as np

from data import PATTERNS, GRID_ROWS, GRID_COLS, get_noisy_pattern
from model import HopfieldNetwork

# Матричная/хакерская тема
ctk.set_appearance_mode("Dark")
COLOR_BG = "#09090B"
COLOR_PANEL = "#18181B"
COLOR_ACCENT = "#10B981"  # Изумрудный/Терминальный
COLOR_TEXT = "#FAFAFA"
CMAP_MATRIX = ListedColormap(['#18181B', '#10B981'])  # Цвета пикселей [-1, 1]


class HopfieldApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Нейронная сеть Хопфилда | Вариант 12")
        self.geometry("1150x700")
        self.configure(fg_color=COLOR_BG)

        # Инициализация и обучение
        self.net = HopfieldNetwork(GRID_ROWS * GRID_COLS)
        self.net.train(list(PATTERNS.values()))

        self.chart_canvas = None
        self.setup_ui()

    def setup_ui(self):
        # Вкладки (Одиночный тест / Пакетный анализ)
        self.tabview = ctk.CTkTabview(
            self, fg_color=COLOR_PANEL, text_color=COLOR_TEXT,
            segmented_button_selected_color=COLOR_ACCENT,
            segmented_button_selected_hover_color="#059669"
        )
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        tab_visual = self.tabview.add("Визуальное распознавание")
        tab_stats = self.tabview.add("Анализ устойчивости (Пакетный тест)")

        self.setup_visual_tab(tab_visual)
        self.setup_stats_tab(tab_stats)

    def setup_visual_tab(self, parent):
        # Панель управления
        control_frame = ctk.CTkFrame(parent, fg_color="transparent")
        control_frame.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(control_frame, text="Настройки паттерна", font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=(0, 20))

        self.letter_var = ctk.StringVar(value="Буква 'П'")
        for letter in PATTERNS.keys():
            ctk.CTkRadioButton(
                control_frame, text=letter, variable=self.letter_var, value=letter,
                fg_color=COLOR_ACCENT, hover_color="#059669"
            ).pack(anchor="w", pady=10)

        self.lbl_noise = ctk.CTkLabel(control_frame, text="Зашумление: 20%", text_color="#A1A1AA")
        self.lbl_noise.pack(pady=(30, 5))

        self.slider_noise = ctk.CTkSlider(
            control_frame, from_=0, to=100, number_of_steps=20,
            button_color=COLOR_ACCENT, progress_color=COLOR_ACCENT,
            command=lambda v: self.lbl_noise.configure(text=f"Зашумление: {int(v)}%")
        )
        self.slider_noise.set(20)
        self.slider_noise.pack(fill="x")

        ctk.CTkButton(
            control_frame, text="Распознать образ", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_ACCENT, text_color="#000000", hover_color="#059669", height=40,
            command=self.run_visual_test
        ).pack(pady=40, fill="x")

        # Холст для отрисовки матриц
        self.canvas_visual = ctk.CTkFrame(parent, fg_color="transparent")
        self.canvas_visual.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        self.run_visual_test()

    def setup_stats_tab(self, parent):
        control_frame = ctk.CTkFrame(parent, fg_color="transparent")
        control_frame.pack(side="top", fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            control_frame, justify="left",
            text="Пакетный тест: прогон 10 тестовых образов для каждого класса на каждом уровне зашумления.\n"
                 "Уровни шума: 10%, 20%, 30%, 35%, 40%, 45%, 50%, 60%, 70%, 80%, 90%, 100%."
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            control_frame, text="Запустить пакетный тест", font=ctk.CTkFont(weight="bold"),
            fg_color="#3B82F6", hover_color="#2563EB", command=self.run_batch_test
        ).pack(side="right", padx=10)

        self.canvas_stats = ctk.CTkFrame(parent, fg_color="transparent")
        self.canvas_stats.pack(side="bottom", fill="both", expand=True, padx=20, pady=10)

    def draw_grid(self, ax, data, title):
        ax.imshow(data.reshape(GRID_ROWS, GRID_COLS), cmap=CMAP_MATRIX, vmin=-1, vmax=1)
        ax.set_title(title, color=COLOR_TEXT, pad=15)
        ax.set_xticks(np.arange(-0.5, GRID_COLS, 1))
        ax.set_yticks(np.arange(-0.5, GRID_ROWS, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='#3F3F46', linestyle='-', linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_color('#3F3F46')
            spine.set_linewidth(2)

    def run_visual_test(self):
        letter = self.letter_var.get()
        noise = int(self.slider_noise.get())

        original = PATTERNS[letter]
        noisy = get_noisy_pattern(original, noise)
        recovered = self.net.predict(noisy)

        # Оценка совпадения
        accuracy = np.mean(original == recovered) * 100

        for w in self.canvas_visual.winfo_children():
            w.destroy()

        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=100)
        fig.patch.set_facecolor(COLOR_PANEL)

        self.draw_grid(ax1, original, f"Эталон ({letter})")
        self.draw_grid(ax2, noisy, f"Искаженный ({noise}% шума)")
        self.draw_grid(ax3, recovered, f"Восстановленный\nТочность: {accuracy:.0f}%")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_visual)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_batch_test(self):
        noise_levels = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        n_tests = 10
        results = {name: [] for name in PATTERNS.keys()}

        for name, original in PATTERNS.items():
            for noise in noise_levels:
                success_count = 0
                for _ in range(n_tests):
                    noisy = get_noisy_pattern(original, noise)
                    recovered = self.net.predict(noisy)
                    if np.array_equal(original, recovered):
                        success_count += 1
                results[name].append((success_count / n_tests) * 100)

        for w in self.canvas_stats.winfo_children():
            w.destroy()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor(COLOR_PANEL)
        ax.set_facecolor(COLOR_BG)

        colors = ['#10B981', '#3B82F6', '#F43F5E']
        for (name, acc_history), color in zip(results.items(), colors):
            ax.plot(noise_levels, acc_history, marker='o', linewidth=2, color=color, label=name)

        ax.set_title("Устойчивость сети к зашумлению (10 прогонов на точку)", color=COLOR_TEXT)
        ax.set_xlabel("Процент зашумления (%)", color="#A1A1AA")
        ax.set_ylabel("Вероятность успешного распознавания (%)", color="#A1A1AA")
        ax.set_xticks(noise_levels)
        ax.set_ylim(-5, 105)
        ax.grid(color='#27272A', linestyle='--')
        ax.legend(facecolor=COLOR_PANEL, edgecolor='#3F3F46')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_stats)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = HopfieldApp()
    app.mainloop()