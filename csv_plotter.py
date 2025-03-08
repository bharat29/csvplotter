import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QLineEdit, QMainWindow, QGridLayout, 
                            QComboBox, QMenu, QHBoxLayout, QScrollArea)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QMouseEvent, QAction
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


class FullScreenPlot(QMainWindow):
    def __init__(self, figure, ax_index, calculated_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Full-Screen Plot {ax_index}")
        self.setGeometry(100, 100, 1200, 800)
        self.figure = figure
        self.ax_index = ax_index
        self.calculated_data = calculated_data
        self.parent_widget = parent
        
        main_layout = QVBoxLayout()
        
        # Determine the root CSVPlotter parent
        root_parent = parent
        while isinstance(root_parent, FullScreenPlot) and root_parent.parent_widget:
            root_parent = root_parent.parent_widget
        
        if ax_index != "All":
            control_layout = QHBoxLayout()
            self.x_scale_combo = QComboBox()
            self.y_scale_combo = QComboBox()
            self.x_min_input = QLineEdit()
            self.x_max_input = QLineEdit()
            self.y_min_input = QLineEdit()
            self.y_max_input = QLineEdit()
            
            self.x_scale_combo.addItems(['Linear', 'Log'])
            self.y_scale_combo.addItems(['Linear', 'Log'])
            self.x_min_input.setPlaceholderText("X min")
            self.x_max_input.setPlaceholderText("X max")
            self.y_min_input.setPlaceholderText("Y min")
            self.y_max_input.setPlaceholderText("Y max")
            
            self.x_scale_combo.currentTextChanged.connect(self.update_scales)
            self.y_scale_combo.currentTextChanged.connect(self.update_scales)
            self.x_min_input.textChanged.connect(self.update_ranges)
            self.x_max_input.textChanged.connect(self.update_ranges)
            self.y_min_input.textChanged.connect(self.update_ranges)
            self.y_max_input.textChanged.connect(self.update_ranges)
            
            control_layout.addWidget(QLabel("X Scale:"))
            control_layout.addWidget(self.x_scale_combo)
            control_layout.addWidget(QLabel("Y Scale:"))
            control_layout.addWidget(self.y_scale_combo)
            control_layout.addWidget(self.x_min_input)
            control_layout.addWidget(self.x_max_input)
            control_layout.addWidget(self.y_min_input)
            control_layout.addWidget(self.y_max_input)

            # Add unit selection for individual plots
            self.time_unit_combo = QComboBox()
            self.time_unit_combo.addItems(['s', 'min', 'hour'])
            self.time_unit_combo.setCurrentText(root_parent.time_unit_combo.currentText())
            self.time_unit_combo.currentTextChanged.connect(self.update_plot)
            control_layout.addWidget(QLabel("Time Unit:"))
            control_layout.addWidget(self.time_unit_combo)

            self.diameter_unit_combo = QComboBox()
            self.diameter_unit_combo.addItems(['mm', 'μm'])
            self.diameter_unit_combo.setCurrentText(root_parent.diameter_unit_combo.currentText())
            self.diameter_unit_combo.currentTextChanged.connect(self.update_plot)
            control_layout.addWidget(QLabel("Diameter Unit:"))
            control_layout.addWidget(self.diameter_unit_combo)

            self.strain_rate_unit_combo = QComboBox()
            self.strain_rate_unit_combo.addItems(['1/s', '1/min', '1/hour', 'μm/s', 'μm/min', 'μm/hour'])
            self.strain_rate_unit_combo.setCurrentText(root_parent.strain_rate_unit_combo.currentText())
            self.strain_rate_unit_combo.currentTextChanged.connect(self.update_plot)
            control_layout.addWidget(QLabel("Strain Rate Unit:"))
            control_layout.addWidget(self.strain_rate_unit_combo)

            main_layout.addLayout(control_layout)
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)
        
        if ax_index == "All":
            self.canvas.mouseDoubleClickEvent = self.double_click_plot

    def show_full_screen(self):
        self.showMaximized()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        export_ascii = QAction("Export to ASCII", self)
        export_pdf = QAction("Export to PDF", self)
        export_jpeg = QAction("Export to JPEG", self)
        export_ascii.triggered.connect(self.export_to_ascii)
        export_pdf.triggered.connect(self.export_to_pdf)
        export_jpeg.triggered.connect(self.export_to_jpeg)
        menu.addAction(export_ascii)
        menu.addAction(export_pdf)
        menu.addAction(export_jpeg)
        menu.exec(self.canvas.mapToGlobal(pos))

    def export_to_ascii(self):
        if self.calculated_data is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ASCII File", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                time_unit = self.time_unit_combo.currentText() if self.ax_index != "All" else self.parent_widget.time_unit_combo.currentText()
                diameter_unit = self.diameter_unit_combo.currentText() if self.ax_index != "All" else self.parent_widget.diameter_unit_combo.currentText()
                strain_rate_unit = self.strain_rate_unit_combo.currentText() if self.ax_index != "All" else self.parent_widget.strain_rate_unit_combo.currentText()
                f.write(f"Time ({time_unit})\tDiameter ({diameter_unit})\tDiametrical Strain (unitless)\tStrain Rate ({strain_rate_unit})\n")
                for t, d, ds, sr in zip(*self.calculated_data):
                    f.write(f"{t}\t{d}\t{ds}\t{sr}\n")

    def export_to_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.figure.savefig(file_path, format='pdf', bbox_inches='tight')

    def export_to_jpeg(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JPEG File", "", "JPEG Files (*.jpg)")
        if file_path:
            self.figure.savefig(file_path, format='jpg', bbox_inches='tight', dpi=300)

    def double_click_plot(self, event: QMouseEvent):
        if not self.figure.axes or self.ax_index != "All":
            return
        
        x, y = event.position().x(), event.position().y()
        y = self.canvas.height() - y
        
        for i, ax in enumerate(self.figure.axes):
            if ax.contains_point((x, y)):
                new_fig = Figure(figsize=(8, 6))
                new_ax = new_fig.add_subplot(111)
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(), line.get_color(), label=line.get_label())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.set_xscale(ax.get_xscale())
                new_ax.set_yscale(ax.get_yscale())
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                new_ax.legend()
                
                new_window = FullScreenPlot(new_fig, f"Plot {i+1}", self.calculated_data, self)
                new_window.show_full_screen()
                break

    def update_scales(self):
        if self.ax_index != "All" and self.figure.axes:
            ax = self.figure.axes[0]
            ax.set_xscale(self.x_scale_combo.currentText().lower())
            ax.set_yscale(self.y_scale_combo.currentText().lower())
            self.canvas.draw()

    def update_ranges(self):
        if self.ax_index != "All" and self.figure.axes:
            ax = self.figure.axes[0]
            try:
                x_min = float(self.x_min_input.text()) if self.x_min_input.text() else None
                x_max = float(self.x_max_input.text()) if self.x_max_input.text() else None
                y_min = float(self.y_min_input.text()) if self.y_min_input.text() else None
                y_max = float(self.y_max_input.text()) if self.y_max_input.text() else None
                
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
                else:
                    ax.relim()  # Recalculate limits based on data
                    ax.autoscale(axis='x')
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
                else:
                    ax.relim()
                    ax.autoscale(axis='y')
            except ValueError:
                ax.relim()
                ax.autoscale()
            self.canvas.draw()

    def update_plot(self):
        if self.ax_index != "All" and self.calculated_data and self.figure.axes:
            ax = self.figure.axes[0]
            time, diameter, diametrical_strain, strain_rate = self.calculated_data
            ax.clear()
            
            time_unit = self.time_unit_combo.currentText()
            diameter_unit = self.diameter_unit_combo.currentText()
            strain_rate_unit = self.strain_rate_unit_combo.currentText()
            
            if self.ax_index == "Plot 1":  # Absolute Diameter
                ax.plot(time, diameter, 'b-', label='Absolute Diameter')
                ax.set_ylabel(f'Diameter ({diameter_unit})')
            elif self.ax_index == "Plot 2":  # Diametrical Strain
                ax.plot(time, diametrical_strain, 'r-', label='Diametrical Strain')
                ax.set_ylabel('Diametrical Strain (unitless)')
            elif self.ax_index == "Plot 3":  # Strain Rate
                ax.plot(time, strain_rate, 'g-', label='Strain Rate')
                ax.set_ylabel(f'Strain Rate ({strain_rate_unit})')
            elif self.ax_index == "Plot 4":  # Creep Rate vs Strain
                ax.plot(diametrical_strain, strain_rate, 'm-', label='Creep Rate vs Strain')
                ax.set_xlabel('Diametrical Strain (unitless)')
                ax.set_ylabel(f'Strain Rate ({strain_rate_unit})')
            
            ax.set_xlabel(f'Time ({time_unit})')
            ax.legend()
            ax.set_xscale(self.x_scale_combo.currentText().lower())
            ax.set_yscale(self.y_scale_combo.currentText().lower())
            
            # Apply custom ranges if set, otherwise autoscale
            self.update_ranges()
            self.canvas.draw()


class CSVPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.csv_file = None
        self.initial_diameter = None
        self.manual_refresh_rate = None
        self.observer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.manual_refresh)
        self.fixed_data = None
        self.calculated_data = None
        self.scales = {
            (0,0): ['linear', 'linear'],
            (0,1): ['linear', 'linear'],
            (1,0): ['linear', 'linear'],
            (1,1): ['linear', 'linear']
        }
        self.ranges = {
            (0,0): [None, None, None, None],
            (0,1): [None, None, None, None],
            (1,0): [None, None, None, None],
            (1,1): [None, None, None, None]
        }

    def initUI(self):
        self.setWindowTitle("Live CSV Plotter")
        self.setGeometry(100, 100, 900, 700)
        main_layout = QVBoxLayout()

        self.label = QLabel("Enter values and press 'Enter' before selecting CSV")
        main_layout.addWidget(self.label)

        self.diameter_input = QLineEdit(self)
        self.diameter_input.setPlaceholderText("Enter Initial Tube Diameter (mm)")
        main_layout.addWidget(self.diameter_input)

        self.manual_refresh_input = QLineEdit(self)
        self.manual_refresh_input.setPlaceholderText("Enter Refresh Interval (s)")
        main_layout.addWidget(self.manual_refresh_input)

        self.btn_enter = QPushButton("Enter")
        self.btn_enter.clicked.connect(self.save_inputs)
        main_layout.addWidget(self.btn_enter)

        self.btn_select = QPushButton("Select CSV File")
        self.btn_select.clicked.connect(self.select_csv)
        main_layout.addWidget(self.btn_select)

        self.btn_auto_refresh = QPushButton("Start Auto Refresh")
        self.btn_auto_refresh.clicked.connect(self.start_auto_refresh)
        main_layout.addWidget(self.btn_auto_refresh)

        self.btn_manual_refresh = QPushButton("Start Manual Refresh")
        self.btn_manual_refresh.clicked.connect(self.start_manual_refresh)
        main_layout.addWidget(self.btn_manual_refresh)

        self.btn_fullscreen = QPushButton("Fullscreen Plot (All)")
        self.btn_fullscreen.clicked.connect(self.open_fullscreen_plot)
        main_layout.addWidget(self.btn_fullscreen)

        # Unit selection controls
        unit_layout = QHBoxLayout()
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(['s', 'min', 'hour'])
        self.time_unit_combo.currentTextChanged.connect(self.manual_refresh)
        unit_layout.addWidget(QLabel("Time Unit:"))
        unit_layout.addWidget(self.time_unit_combo)

        self.diameter_unit_combo = QComboBox()
        self.diameter_unit_combo.addItems(['mm', 'μm'])
        self.diameter_unit_combo.currentTextChanged.connect(self.manual_refresh)
        unit_layout.addWidget(QLabel("Diameter Unit:"))
        unit_layout.addWidget(self.diameter_unit_combo)

        self.strain_rate_unit_combo = QComboBox()
        self.strain_rate_unit_combo.addItems(['1/s', '1/min', '1/hour', 'μm/s', 'μm/min', 'μm/hour'])
        self.strain_rate_unit_combo.currentTextChanged.connect(self.manual_refresh)
        unit_layout.addWidget(QLabel("Strain Rate Unit:"))
        unit_layout.addWidget(self.strain_rate_unit_combo)
        main_layout.addLayout(unit_layout)

        # Scrollable area for plots and controls
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.scale_controls = {}
        self.range_controls = {}
        scale_layout = QGridLayout()
        for i, (pos, title) in enumerate([
            ((0,0), "Absolute Diameter"),
            ((0,1), "Diametrical Strain"),
            ((1,0), "Strain Rate"),
            ((1,1), "Creep Rate vs Strain")
        ]):
            label = QLabel(title)
            x_combo = QComboBox()
            y_combo = QComboBox()
            x_combo.addItems(['Linear', 'Log'])
            y_combo.addItems(['Linear', 'Log'])
            x_combo.currentTextChanged.connect(lambda scale, p=pos: self.update_scale(p, 'x', scale.lower()))
            y_combo.currentTextChanged.connect(lambda scale, p=pos: self.update_scale(p, 'y', scale.lower()))
            self.scale_controls[pos] = (x_combo, y_combo)

            x_min = QLineEdit()
            x_max = QLineEdit()
            y_min = QLineEdit()
            y_max = QLineEdit()
            x_min.setPlaceholderText("X min")
            x_max.setPlaceholderText("X max")
            y_min.setPlaceholderText("Y min")
            y_max.setPlaceholderText("Y max")
            x_min.textChanged.connect(lambda text, p=pos: self.update_range(p, 'x_min', text))
            x_max.textChanged.connect(lambda text, p=pos: self.update_range(p, 'x_max', text))
            y_min.textChanged.connect(lambda text, p=pos: self.update_range(p, 'y_min', text))
            y_max.textChanged.connect(lambda text, p=pos: self.update_range(p, 'y_max', text))
            self.range_controls[pos] = (x_min, x_max, y_min, y_max)

            scale_layout.addWidget(label, i, 0)
            scale_layout.addWidget(QLabel("X Scale:"), i, 1)
            scale_layout.addWidget(x_combo, i, 2)
            scale_layout.addWidget(QLabel("Y Scale:"), i, 3)
            scale_layout.addWidget(y_combo, i, 4)
            scale_layout.addWidget(x_min, i, 5)
            scale_layout.addWidget(x_max, i, 6)
            scale_layout.addWidget(y_min, i, 7)
            scale_layout.addWidget(y_max, i, 8)
        scroll_layout.addLayout(scale_layout)

        self.figure = Figure(figsize=(8, 8))  # Adjusted for better aspect ratio
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)

        scroll_layout.addWidget(self.toolbar)
        scroll_layout.addWidget(self.canvas)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.canvas.mouseDoubleClickEvent = self.double_click_plot

    def save_inputs(self):
        try:
            self.initial_diameter = float(self.diameter_input.text())
            self.manual_refresh_rate = int(float(self.manual_refresh_input.text()) * 1000)
            self.label.setText(f"Values Set: Diameter = {self.initial_diameter} mm, Refresh Rate = {self.manual_refresh_rate / 1000} s")
        except ValueError:
            self.label.setText("Invalid input. Enter numeric values and press 'Enter'.")

    def select_csv(self):
        if self.initial_diameter is None:
            self.label.setText("Set values and press 'Enter' first!")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_file = file_path
            self.label.setText(f"Selected: {file_path}")
            self.event_handler = CSVEventHandler(self.csv_file, self.initial_diameter, self)
            self.event_handler.plot_data()  # Initial plot

    def start_auto_refresh(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        if not self.csv_file:
            self.label.setText("No CSV file selected!")
            return

        self.timer.stop()  # Stop manual refresh if running
        self.event_handler = CSVEventHandler(self.csv_file, self.initial_diameter, self)
        self.observer = Observer()
        folder_path = "/".join(self.csv_file.split("/")[:-1])
        self.observer.schedule(self.event_handler, folder_path, recursive=False)
        self.observer.start()
        self.label.setText(f"Auto-refresh started for: {self.csv_file}")
        self.event_handler.plot_data()  # Initial plot
        print(f"Observer started for {folder_path}")

    def start_manual_refresh(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()

        if self.manual_refresh_rate is None:
            self.label.setText("Set values and press 'Enter' first!")
            return

        if not self.csv_file:
            self.label.setText("No CSV file selected!")
            return

        self.event_handler = CSVEventHandler(self.csv_file, self.initial_diameter, self)
        self.timer.start(self.manual_refresh_rate)
        self.label.setText(f"Manual refresh started (interval: {self.manual_refresh_rate / 1000} s)")
        self.event_handler.plot_data()  # Initial plot

    def manual_refresh(self):
        if self.csv_file and self.event_handler:
            self.event_handler.plot_data()

    def open_fullscreen_plot(self):
        self.fullscreen_window = FullScreenPlot(self.figure, "All", self.calculated_data, self)
        self.fullscreen_window.show_full_screen()

    def double_click_plot(self, event: QMouseEvent):
        if not self.figure.axes:
            return
        
        x, y = event.position().x(), event.position().y()
        y = self.canvas.height() - y
        
        for i, ax in enumerate(self.figure.axes):
            if ax.contains_point((x, y)):
                new_fig = Figure(figsize=(8, 6))
                new_ax = new_fig.add_subplot(111)
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(), line.get_color(), label=line.get_label())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.set_xscale(ax.get_xscale())
                new_ax.set_yscale(ax.get_yscale())
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                new_ax.legend()
                
                self.fullscreen_window = FullScreenPlot(new_fig, f"Plot {i+1}", self.calculated_data, self)
                self.fullscreen_window.show_full_screen()
                break

    def update_scale(self, position, axis, scale):
        idx = 0 if axis == 'x' else 1
        self.scales[position][idx] = scale
        self.manual_refresh()

    def update_range(self, position, axis, value):
        idx = {'x_min': 0, 'x_max': 1, 'y_min': 2, 'y_max': 3}[axis]
        try:
            self.ranges[position][idx] = float(value) if value else None
        except ValueError:
            self.ranges[position][idx] = None
        self.manual_refresh()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        export_action = QAction("Export to ASCII", self)
        export_action.triggered.connect(self.export_to_ascii)
        menu.addAction(export_action)
        menu.exec(self.canvas.mapToGlobal(pos))

    def export_to_ascii(self):
        if self.calculated_data is None:
            self.label.setText("No data available to export!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save ASCII File", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                time_unit = self.time_unit_combo.currentText()
                diameter_unit = self.diameter_unit_combo.currentText()
                strain_rate_unit = self.strain_rate_unit_combo.currentText()
                f.write(f"Time ({time_unit})\tDiameter ({diameter_unit})\tDiametrical Strain (unitless)\tStrain Rate ({strain_rate_unit})\n")
                for t, d, ds, sr in zip(*self.calculated_data):
                    f.write(f"{t}\t{d}\t{ds}\t{sr}\n")
            self.label.setText(f"Data exported to {file_path}")


class CSVEventHandler(FileSystemEventHandler):
    def __init__(self, file_path, initial_diameter, parent):
        self.file_path = file_path
        self.initial_diameter = initial_diameter  # In mm
        self.parent = parent

    def on_modified(self, event):
        if event.src_path == self.file_path:
            print(f"Detected modification: {self.file_path}")
            self.plot_data()

    def plot_data(self):
        self.parent.figure.clear()
        axs = self.parent.figure.subplots(2, 2)
        
        try:
            if self.file_path:
                df = pd.read_csv(self.file_path).dropna()
                print(f"CSV rows: {len(df)}")  # Debug: Check data size
                time = df.iloc[:, 0].values  # Time in seconds (base unit)
                diameter = df.iloc[:, 1].values  # Diameter in mm

                # Smooth diameter
                diameter_smooth = savgol_filter(diameter, 5, 2)

                # Convert time based on selected unit
                time_unit = self.parent.time_unit_combo.currentText()
                if time_unit == 's':
                    time_converted = time
                    time_factor = 1
                elif time_unit == 'min':
                    time_converted = time / 60  # s to min
                    time_factor = 60
                elif time_unit == 'hour':
                    time_converted = time / 3600  # s to hour
                    time_factor = 3600

                # Convert diameter based on selected unit
                diameter_unit = self.parent.diameter_unit_combo.currentText()
                if diameter_unit == 'μm':
                    diameter_converted = diameter_smooth * 1000  # mm to μm
                    initial_diameter_converted = self.initial_diameter * 1000
                else:  # mm
                    diameter_converted = diameter_smooth
                    initial_diameter_converted = self.initial_diameter

                # Calculate diametrical strain (unitless)
                diametrical_strain = (diameter_converted - initial_diameter_converted) / initial_diameter_converted

                # Calculate strain rate and convert based on selected unit
                strain_rate_unit = self.parent.strain_rate_unit_combo.currentText()
                strain_rate = np.gradient(diametrical_strain, time)  # Base strain rate in 1/s
                if strain_rate_unit == '1/s':
                    strain_rate_converted = strain_rate * time_factor  # Adjust for time unit
                elif strain_rate_unit == '1/min':
                    strain_rate_converted = strain_rate * time_factor * 60
                elif strain_rate_unit == '1/hour':
                    strain_rate_converted = strain_rate * time_factor * 3600
                elif strain_rate_unit == 'μm/s':
                    strain_rate_converted = strain_rate * time_factor * 1000
                elif strain_rate_unit == 'μm/min':
                    strain_rate_converted = strain_rate * time_factor * 1000 * 60
                elif strain_rate_unit == 'μm/hour':
                    strain_rate_converted = strain_rate * time_factor * 1000 * 3600

                # Store calculated data for export
                self.parent.calculated_data = (time_converted, diameter_converted, diametrical_strain, strain_rate_converted)

                # Plot data
                axs[0, 0].plot(time_converted, diameter_converted, 'b-', label='Absolute Diameter')
                axs[0, 1].plot(time_converted, diametrical_strain, 'r-', label='Diametrical Strain')
                axs[1, 0].plot(time_converted, strain_rate_converted, 'g-', label='Strain Rate')
                axs[1, 1].plot(diametrical_strain, strain_rate_converted, 'm-', label='Creep Rate vs Strain')

                for pos, ax in [((0,0), axs[0,0]), ((0,1), axs[0,1]), 
                              ((1,0), axs[1,0]), ((1,1), axs[1,1])]:
                    x_scale, y_scale = self.parent.scales[pos]
                    x_min, x_max, y_min, y_max = self.parent.ranges[pos]
                    ax.set_xscale(x_scale)
                    ax.set_yscale(y_scale)
                    if x_min is not None and x_max is not None:
                        ax.set_xlim(x_min, x_max)
                    if y_min is not None and y_max is not None:
                        ax.set_ylim(y_min, y_max)

            # Set axis labels with units
            axs[0, 0].set_xlabel(f'Time ({time_unit})')
            axs[0, 0].set_ylabel(f'Diameter ({diameter_unit})')
            axs[0, 1].set_xlabel(f'Time ({time_unit})')
            axs[0, 1].set_ylabel('Diametrical Strain (unitless)')
            axs[1, 0].set_xlabel(f'Time ({time_unit})')
            axs[1, 0].set_ylabel(f'Strain Rate ({strain_rate_unit})')
            axs[1, 1].set_xlabel('Diametrical Strain (unitless)')
            axs[1, 1].set_ylabel(f'Strain Rate ({strain_rate_unit})')

            for ax in axs.flat:
                ax.legend()

            plt.tight_layout(pad=1.0)
            QTimer.singleShot(0, self.parent.canvas.draw)  # Thread-safe GUI update
        except Exception as e:
            print(f"Error in plot_data: {e}")
            self.parent.label.setText(f"Plotting error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVPlotter()
    window.show()
    sys.exit(app.exec())