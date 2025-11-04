import sys
import os
import io
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QTableView, QTextEdit, QSplitter
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.use("Agg")  # безопасный backend для рендеринга
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Вспомогательные виджеты ----------

class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)


class PandasModel(QtCore.QAbstractTableModel):
    """Модель для отображения pandas.DataFrame в QTableView."""
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])

    def set_df(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()


# ---------- Главное окно ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализация данных — PyQt")
        self.resize(1200, 800)

        # ✅ Подключение к SQLite (совместимо с разными версиями SQLAlchemy)
        from sqlalchemy import inspect
        self.db_path = os.path.abspath("app.db")
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.inspector = inspect(self.engine)

        # Лог
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Меню
        self._make_menu()

        # Вкладки
        self.tab_stats = self._build_tab_stats()
        self.tabs.addTab(self.tab_stats, "1) Статистика")

        self.tab_pairs = self._build_tab_pairs()
        self.tabs.addTab(self.tab_pairs, "2) Корреляции")

        self.tab_heat = self._build_tab_heatmap()
        self.tabs.addTab(self.tab_heat, "3) Тепловая карта")

        self.tab_line = self._build_tab_line()
        self.tabs.addTab(self.tab_line, "4) Линейный график")

        self.tab_log = self._build_tab_log()
        self.tabs.addTab(self.tab_log, "5) Ход работы (лог)")

        self._refresh_tables_everywhere()
        self._log("Приложение запущено.")

    # ---------- Меню ----------
    def _make_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")

        open_csv = QAction("Загрузить CSV", self)
        open_csv.triggered.connect(self.load_csv)
        file_menu.addAction(open_csv)

        open_excel = QAction("Загрузить Excel", self)
        open_excel.triggered.connect(self.load_excel)
        file_menu.addAction(open_excel)

        exit_act = QAction("Выход", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    # ---------- Таб 1: Статистика ----------
    def _build_tab_stats(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_table_stats = QComboBox()
        btn_refresh = QPushButton("Обновить статистику")
        btn_refresh.clicked.connect(self._update_stats)
        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_table_stats, 1)
        top.addWidget(btn_refresh)

        splitter = QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)

        # верх: предпросмотр данных
        self.tbl_preview = QTableView()
        self.preview_model = PandasModel(pd.DataFrame())
        self.tbl_preview.setModel(self.preview_model)
        splitter.addWidget(self.tbl_preview)

        # низ: текстовая статистика
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        splitter.addWidget(self.stats_text)

        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        return w

    def _update_stats(self):
        table = self.cb_table_stats.currentText().strip()
        if not table:
            QMessageBox.warning(self, "Нет таблицы", "Выберите таблицу.")
            return
        try:
            # ✅ читаем таблицу напрямую через sqlite
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения", str(e))
            return


        # предпросмотр
        self.preview_model.set_df(df.head(200))

        # статистика
        buf = io.StringIO()
        print(f"Статистика по таблице: {table}", file=buf)
        print(f"Строк: {len(df)}; Столбцов: {df.shape[1]}", file=buf)
        print("\nТипы данных:", file=buf)
        print(df.dtypes.to_string(), file=buf)

        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            print("\nЧисловая describe():", file=buf)
            print(num_df.describe().to_string(), file=buf)

            print("\nПропуски (числовые):", file=buf)
            print(num_df.isna().sum().to_string(), file=buf)
        else:
            print("\nЧисловых столбцов не найдено.", file=buf)

        self.stats_text.setPlainText(buf.getvalue())
        self._log(f"Обновлена статистика по таблице '{table}'.")


    # ---------- Таб 2: Корреляции (pairplot) ----------
    def _build_tab_pairs(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_table_pairs = QComboBox()
        self.list_num_cols = QListWidget()
        self.list_num_cols.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_num_cols.setMinimumHeight(120)

        btn_load_cols = QPushButton("Загрузить столбцы")
        btn_load_cols.clicked.connect(self._load_numeric_cols_for_pairs)
        btn_plot = QPushButton("Построить графики корреляции")
        btn_plot.clicked.connect(self._plot_pairplot)

        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_table_pairs, 1)
        top.addWidget(btn_load_cols)
        top.addWidget(btn_plot)

        # контейнер под canvas, чтобы можно было легко подменять виджет
        self.pairs_canvas_container = QWidget()
        self.pairs_canvas_layout = QVBoxLayout(self.pairs_canvas_container)
        self.pairs_canvas_layout.setContentsMargins(0, 0, 0, 0)

        # первый пустой canvas
        self.canvas_pairs = MplCanvas(width=7, height=5)
        self.pairs_canvas_layout.addWidget(self.canvas_pairs)

        layout.addLayout(top)
        layout.addWidget(QLabel("Выберите числовые столбцы (Ctrl/Shift для множественного выбора):"))
        layout.addWidget(self.list_num_cols)
        layout.addWidget(self.pairs_canvas_container, 1)
        return w


    def _load_numeric_cols_for_pairs(self):
        tbl = self.cb_table_pairs.currentText().strip()
        if not tbl:
            QMessageBox.warning(self, "Нет таблицы", "Выберите таблицу.")
            return
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.list_num_cols.clear()
        for c in num_cols:
            item = QListWidgetItem(c)
            self.list_num_cols.addItem(item)
        self._log(f"Загружены числовые столбцы для pairplot: {len(num_cols)} шт.")

    def _plot_pairplot(self):
        import sqlite3
        tbl = self.cb_table_pairs.currentText().strip()
        if not tbl:
            QMessageBox.warning(self, "Нет таблицы", "Выберите таблицу.")
            return

        # читаем таблицу из sqlite (устойчиво на любых версиях SQLAlchemy)
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения таблицы", str(e))
            return

        selected = [i.text() for i in self.list_num_cols.selectedItems()]
        if len(selected) < 2:
            QMessageBox.warning(self, "Мало столбцов", "Выберите хотя бы 2 числовых столбца.")
            return
        if len(selected) > 6:
            QMessageBox.warning(self, "Слишком много", "Выберите не более 6 столбцов (для скорости).")
            return

        # строим pairplot — он вернёт свою figure
        try:
            g = sns.pairplot(df[selected], corner=True, plot_kws=dict(s=25, linewidth=0))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка построения графика", str(e))
            return

        # подменяем canvas внутри контейнера на новый, созданный из figure seaborn
        # 1) удалить старый canvas
        while self.pairs_canvas_layout.count():
            item = self.pairs_canvas_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        # 2) создать новый FigureCanvas на основе g.fig
        new_canvas = FigureCanvas(g.fig)
        self.pairs_canvas_layout.addWidget(new_canvas)
        new_canvas.draw()

        # закрыть лишние ссылки pyplot
        plt.close(g.fig)

        self._log(f"Построен pairplot по столбцам: {', '.join(selected)}.")




        g = sns.pairplot(df[selected], corner=True, plot_kws=dict(s=25, linewidth=0))
        # перерисуем в наш canvas:
        self.canvas_pairs.fig.clf()
        self.canvas_pairs.fig = g.fig  # перенос figure
        self.canvas_pairs.draw()
        plt.close(g.fig)
        self._log(f"Построен pairplot по столбцам: {', '.join(selected)}.")

    # ---------- Таб 3: Тепловая карта ----------
    def _build_tab_heatmap(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.cb_table_heat = QComboBox()
        btn_plot = QPushButton("Построить тепловую карту корреляций")
        btn_plot.clicked.connect(self._plot_heatmap)

        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_table_heat, 1)
        top.addWidget(btn_plot)

        self.canvas_heat = MplCanvas(width=7, height=5)
        layout.addLayout(top)
        layout.addWidget(self.canvas_heat, 1)
        return w

    def _plot_heatmap(self):
        tbl = self.cb_table_heat.currentText().strip()
        if not tbl:
            QMessageBox.warning(self, "Нет таблицы", "Выберите таблицу.")
            return
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            QMessageBox.information(self, "Недостаточно данных", "Нужно минимум 2 числовых столбца.")
            return

        corr = num_df.corr(numeric_only=True)
        self.canvas_heat.fig.clf()
        ax = self.canvas_heat.fig.add_subplot(111)
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        ax.set_title("Корреляционная матрица")
        self.canvas_heat.draw()
        self._log("Построена тепловая карта корреляций.")

    # ---------- Таб 4: Линейный график ----------
    # ---------- Таб 4: Линейный график ----------
    def _build_tab_line(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Верхняя панель управления
        top = QHBoxLayout()
        self.cb_table_line = QComboBox()
        self.cb_col_line = QComboBox()

        btn_load_cols = QPushButton("Загрузить столбцы")
        btn_load_cols.clicked.connect(self._load_cols_for_line)
        btn_plot = QPushButton("Построить график")
        btn_plot.clicked.connect(self._plot_line)

        top.addWidget(QLabel("Таблица:"))
        top.addWidget(self.cb_table_line, 1)
        top.addWidget(QLabel("Столбец (Y):"))
        top.addWidget(self.cb_col_line, 1)
        top.addWidget(btn_load_cols)
        top.addWidget(btn_plot)

        # Полотно для графика
        self.canvas_line = MplCanvas(width=7, height=5)
        layout.addLayout(top)
        layout.addWidget(self.canvas_line, 1)
        return w


    def _load_cols_for_line(self):
        """Загружает числовые столбцы для выбора"""
        tbl = self.cb_table_line.currentText().strip()
        if not tbl:
            QMessageBox.warning(self, "Нет таблицы", "Выберите таблицу.")
            return

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cb_col_line.clear()
        self.cb_col_line.addItems(num_cols)

        self._log(f"Загружены числовые столбцы для таблицы '{tbl}'.")


    def _plot_line(self):
        """Строит линейный график одного столбца относительно индекса наблюдения"""
        tbl = self.cb_table_line.currentText().strip()
        col_y = self.cb_col_line.currentText().strip()

        if not tbl or not col_y:
            QMessageBox.warning(self, "Нет данных", "Выберите таблицу и столбец.")
            return

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)

        y = pd.to_numeric(df[col_y], errors="coerce")
        x = np.arange(len(y))
        mean_val = np.nanmean(y)

        # Очистка и построение
        self.canvas_line.fig.clf()
        ax = self.canvas_line.fig.add_subplot(111)

        ax.plot(x, y, marker='o', markersize=3, linewidth=0.8, alpha=0.8)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.2,
                label=f"Среднее: {mean_val:.2f}")
        ax.legend(loc='upper right')

        ax.set_xlabel("Индекс наблюдения")
        ax.set_ylabel(col_y)
        ax.set_title(f"Линейный график: {col_y}")
        ax.grid(True, which='both', alpha=0.3)

        self.canvas_line.draw()
        self._log(f"Построен график '{col_y}' относительно индекса наблюдения.")


    # ---------- Таб 5: Лог ----------
    def _build_tab_log(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(self.log_widget)
        return w

    def _log(self, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_widget.append(f"[{ts}] {text}")

    # ---------- Загрузка данных ----------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV-файл", "", "CSV (*.csv);;Все файлы (*.*)")
        if not path:
            return
        table, ok = QtWidgets.QInputDialog.getText(
            self, "Имя таблицы", "Как назвать таблицу в БД?",
            text=os.path.splitext(os.path.basename(path))[0]
        )
        if not ok or not table:
            return
        try:
            df = pd.read_csv(path)
            # ✅ через соединение (устойчиво на разных версиях)
            with self.engine.begin() as conn:
                df.to_sql(table, conn, if_exists="replace", index=False)
            self._refresh_tables_everywhere()
            self._log(f"CSV загружен: '{path}' → таблица '{table}' ({len(df)} строк).")
            QMessageBox.information(self, "Успех", f"Данные загружены в таблицу '{table}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки CSV", str(e))

    def load_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите Excel-файл", "", "Excel (*.xlsx *.xls);;Все файлы (*.*)")
        if not path:
            return
        sheet, ok = QtWidgets.QInputDialog.getText(self, "Лист Excel", "Имя листа (пусто — первый):", text="")
        table, ok2 = QtWidgets.QInputDialog.getText(
            self, "Имя таблицы", "Как назвать таблицу в БД?",
            text=os.path.splitext(os.path.basename(path))[0]
    )
        if not ok2 or not table:
            return
        try:
            df = pd.read_excel(path, sheet_name=sheet if sheet else None)
            if isinstance(df, dict):  # если много листов — берём первый
                df = list(df.values())[0]

        # ✅ используем прямое подключение к sqlite для избежания ошибки "cursor"
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table, conn, if_exists="replace", index=False)

            self._refresh_tables_everywhere()
            self._log(f"Excel загружен: '{path}' → таблица '{table}' ({len(df)} строк).")
            QMessageBox.information(self, "Успех", f"Данные загружены в таблицу '{table}'.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки Excel", str(e))



    def _refresh_tables_everywhere(self):
        from sqlalchemy import inspect
        insp = inspect(self.engine)
        tables = insp.get_table_names()

        for cb in [self.cb_table_stats, self.cb_table_pairs, self.cb_table_heat, self.cb_table_line]:
            cb.clear()
            cb.addItems(tables)

        if tables:
            # автообновление статистики при наличии таблиц
            self.cb_table_stats.setCurrentIndex(0)
            self._update_stats()


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
