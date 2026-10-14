import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from monte_carlo import run_simulation

# Настройка глобального стиля
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Цветовая палитра карточек и акцентов
COLOR_BG = "#0F172A"          # Slate 900
COLOR_CARD = "#1E293B"        # Slate 800
COLOR_ACCENT = "#38BDF8"      # Sky 400
COLOR_ACCENT_HOVER = "#0284C7"
COLOR_SUCCESS = "#4ADE80"     # Green 400
COLOR_TEXT = "#F8FAFC"
COLOR_MUTED = "#94A3B8"
COLOR_BORDER = "#334155"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Монте-Карло | Оптимизация входного контроля (Вариант 12)")
        self.geometry("980x740")
        self.minsize(850, 650)
        self.configure(fg_color=COLOR_BG)

        self.canvas = None
        self.entries = {}
        self.create_layout()

    def create_layout(self):
        # Верхняя плашка заголовка
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=25, pady=(20, 10))

        title = ctk.CTkLabel(
            header_frame,
            text="Оптимизация затрат на входной контроль",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLOR_TEXT
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Моделирование методом Монте-Карло с генератором LFSR (Полином №12)",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_MUTED
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Основная рабочая область (2 колонки)
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=25, pady=(5, 20))
        main_content.columnconfigure(0, weight=0, minsize=320)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=1)

        # Левая колонка: параметры и управление
        left_panel = ctk.CTkFrame(main_content, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        lbl_settings = ctk.CTkLabel(
            left_panel,
            text="Параметры варианта 12",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLOR_TEXT
        )
        lbl_settings.pack(anchor="w", padx=20, pady=(20, 15))

        # Ввод параметров Варианта 12 по умолчанию
        inputs = [
            ("A - Доля брака (%):", "7", "a"),
            ("B - Стоимость контроля (руб):", "5", "b"),
            ("C - Стоимость подгонки (руб):", "55", "c"),
            ("D - Стоимость замены (руб):", "95", "d"),
            ("N - Количество испытаний:", "10000", "n"),
        ]

        for text, default_val, key in inputs:
            self.add_input_field(left_panel, text, default_val, key)

        self.btn_run = ctk.CTkButton(
            left_panel,
            text="Запустить расчет",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#0F172A",
            corner_radius=10,
            height=42,
            command=self.calculate
        )
        self.btn_run.pack(fill="x", padx=20, pady=(20, 15))

        # Карточка для отображения оптимума
        self.card_result = ctk.CTkFrame(left_panel, fg_color="#0F172A", corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        self.card_result.pack(fill="x", padx=20, pady=(5, 20))

        self.lbl_res_title = ctk.CTkLabel(
            self.card_result,
            text="Итог оптимизации:",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_MUTED
        )
        self.lbl_res_title.pack(anchor="w", padx=15, pady=(12, 2))

        self.lbl_result = ctk.CTkLabel(
            self.card_result,
            text="Ожидание запуска...",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLOR_TEXT,
            wraplength=260,
            justify="left"
        )
        self.lbl_result.pack(anchor="w", padx=15, pady=(0, 12))

        # Правая колонка: вкладки с результатами
        right_panel = ctk.CTkFrame(main_content, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        right_panel.grid(row=0, column=1, sticky="nsew")

        self.tabview = ctk.CTkTabview(
            right_panel,
            fg_color="transparent",
            segmented_button_selected_color=COLOR_ACCENT,
            segmented_button_selected_hover_color=COLOR_ACCENT_HOVER,
            segmented_button_unselected_color="#0F172A",
            text_color=COLOR_TEXT
        )
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)

        self.tabview.add("График затрат")
        self.tabview.add("Сводная таблица")

        # Настройка таблицы
        self.table_frame = ctk.CTkScrollableFrame(self.tabview.tab("Сводная таблица"), fg_color="transparent")
        self.table_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def add_input_field(self, parent, label_text, default_val, key):
        lbl = ctk.CTkLabel(
            parent,
            text=label_text,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_MUTED
        )
        lbl.pack(anchor="w", padx=20, pady=(4, 2))

        entry = ctk.CTkEntry(
            parent,
            fg_color="#0F172A",
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
            corner_radius=8,
            height=34
        )
        entry.insert(0, default_val)
        entry.pack(fill="x", padx=20, pady=(0, 8))
        self.entries[key] = entry

    def calculate(self):
        # Валидация входных данных
        try:
            a = float(self.entries["a"].get())
            b = float(self.entries["b"].get())
            c = float(self.entries["c"].get())
            d = float(self.entries["d"].get())
            n = int(self.entries["n"].get())
        except ValueError:
            self.lbl_result.configure(text="Ошибка ввода! Проверьте числа.", text_color="#F87171")
            return

        self.lbl_result.configure(text="Выполняется симуляция...", text_color=COLOR_TEXT)
        self.update()

        results = run_simulation(a, b, c, d, n)

        x_vals = [r[0] for r in results]
        y_vals = [r[1] for r in results]
        min_cost = min(y_vals)
        optimal_proc = x_vals[y_vals.index(min_cost)]

        # Обновление карточки результата
        self.lbl_result.configure(
            text=f"Оптимальный контроль: {optimal_proc}%\nМинимум затрат: {min_cost:,.2f} руб.",
            text_color=COLOR_SUCCESS
        )

        # Заполнение таблицы
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        h1 = ctk.CTkLabel(self.table_frame, text="Доля контроля", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_ACCENT)
        h1.grid(row=0, column=0, padx=25, pady=8, sticky="w")
        h2 = ctk.CTkLabel(self.table_frame, text="Суммарные затраты", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_ACCENT)
        h2.grid(row=0, column=1, padx=25, pady=8, sticky="e")
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.columnconfigure(1, weight=1)

        for i, (proc, cost) in enumerate(results, start=1):
            is_opt = (cost == min_cost)
            row_color = COLOR_SUCCESS if is_opt else COLOR_TEXT

            txt_proc = f"★ {proc}% (Оптимум)" if is_opt else f"{proc}%"
            lbl_p = ctk.CTkLabel(self.table_frame, text=txt_proc, font=ctk.CTkFont(size=13), text_color=row_color)
            lbl_p.grid(row=i, column=0, padx=25, pady=3, sticky="w")

            lbl_c = ctk.CTkLabel(self.table_frame, text=f"{cost:,.2f} руб.", font=ctk.CTkFont(size=13), text_color=row_color)
            lbl_c.grid(row=i, column=1, padx=25, pady=3, sticky="e")

        # Построение графика в стиле тёмной темы
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
        fig.patch.set_facecolor(COLOR_CARD)
        ax.set_facecolor("#0F172A")

        ax.plot(x_vals, y_vals, marker='o', markersize=5, color=COLOR_ACCENT, linewidth=2, label='Суммарные затраты')
        ax.plot(optimal_proc, min_cost, marker='*', markersize=14, color="#FBBF24", linestyle='None', label=f'Минимум ({optimal_proc}%)')

        ax.set_title("Кривая совокупных затрат", color=COLOR_TEXT, fontsize=12, pad=10)
        ax.set_xlabel("Процент входного контроля (%)", color=COLOR_MUTED, fontsize=10)
        ax.set_ylabel("Затраты (руб)", color=COLOR_MUTED, fontsize=10)

        ax.tick_params(colors=COLOR_MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLOR_BORDER)

        ax.grid(True, linestyle='--', color=COLOR_BORDER, alpha=0.6)
        legend = ax.legend(facecolor=COLOR_CARD, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT)

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("График затрат"))
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)


if __name__ == "__main__":
    app = App()
    app.mainloop()