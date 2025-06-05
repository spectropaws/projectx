import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import pyqtSlot
from projectx_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect buttons to methods
        self.ui.generateButton.clicked.connect(self.on_generate_clicked)
        self.ui.apply_button.clicked.connect(self.on_apply_clicked)
        self.ui.cancel_button.clicked.connect(self.on_cancel_clicked)

        # Monitor changes to new config text
        self.ui.newConfigText.textChanged.connect(self.on_config_changed)

        # Initial setup
        self.update_button_states()

    @pyqtSlot()
    def on_generate_clicked(self):
        prompt = self.ui.promptInput.toPlainText()
        if not prompt.strip():
            self.ui.llmOutput.setPlainText("Prompt is empty. Please enter a prompt.")
            return

        # Simulate LLM generation (replace this with your actual LLM integration)
        generated_config = f"# Generated config based on prompt:\n{prompt}"
        self.ui.llmOutput.setPlainText(generated_config)
        self.ui.newConfigText.setPlainText(generated_config)

        self.update_button_states()

    @pyqtSlot()
    def on_apply_clicked(self):
        new_config = self.ui.newConfigText.toPlainText()
        self.ui.oldConfigText.setPlainText(new_config)
        self.update_button_states()

    @pyqtSlot()
    def on_cancel_clicked(self):
        self.ui.newConfigText.clear()
        self.ui.llmOutput.clear()
        self.update_button_states()

    @pyqtSlot()
    def on_config_changed(self):
        self.update_button_states()

    def update_button_states(self):
        old_config = self.ui.oldConfigText.toPlainText().strip()
        new_config = self.ui.newConfigText.toPlainText().strip()

        enable = bool(new_config) and (new_config != old_config)
        self.ui.apply_button.setEnabled(enable)
        self.ui.cancel_button.setEnabled(enable)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
